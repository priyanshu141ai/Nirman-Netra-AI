"""Versioned baseline artifact, ONNX export, and strict inference loader."""

import json
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument, RuntimeException
from pydantic import ValidationError

from nirman_netra.exceptions import (
    ArtifactValidationError,
    InferenceValidationError,
    StorageError,
)
from nirman_netra.segmentation.contracts import (
    InputSchema,
    ModelArtifactMetadata,
    SegmentationMetrics,
)
from nirman_netra.segmentation.model import LinearSegmentationModel
from nirman_netra.utils import file_content_hash

METADATA_FILENAME = "model-artifact.json"
WEIGHTS_FILENAME = "weights.npy"
ONNX_FILENAME = "model.onnx"


def _export_onnx(model: LinearSegmentationModel, schema: InputSchema, path: Path) -> None:
    mean = np.asarray(model.normalization.mean, dtype=np.float32)
    std = np.asarray(model.normalization.std, dtype=np.float32)
    weights = model.weights.reshape(schema.channels, 1).astype(np.float32)
    bias = np.asarray([model.bias], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["image", "mean"], ["centered"]),
            helper.make_node("Div", ["centered", "std"], ["normalized"]),
            helper.make_node("MatMul", ["normalized", "weights"], ["logits_unbiased"]),
            helper.make_node("Add", ["logits_unbiased", "bias"], ["logits"]),
            helper.make_node("Sigmoid", ["logits"], ["probabilities"]),
        ],
        "nirman_netra_linear_segmentation",
        [
            helper.make_tensor_value_info(
                "image",
                TensorProto.FLOAT,
                ["batch", schema.height, schema.width, schema.channels],
            )
        ],
        [
            helper.make_tensor_value_info(
                "probabilities",
                TensorProto.FLOAT,
                ["batch", schema.height, schema.width, 1],
            )
        ],
        [
            numpy_helper.from_array(mean, "mean"),
            numpy_helper.from_array(std, "std"),
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    exported = helper.make_model(
        graph,
        producer_name="nirman-netra",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(exported)
    path.write_bytes(exported.SerializeToString())


def save_model_artifact(
    model: LinearSegmentationModel,
    output_directory: Path,
    *,
    model_version: str,
    training_dataset_version: str,
    input_schema: InputSchema,
    evaluation_metrics: SegmentationMetrics,
) -> ModelArtifactMetadata:
    """Save deterministic weights, an ONNX graph, and machine-readable metadata."""

    if len(model.weights) != input_schema.channels:
        raise ArtifactValidationError("model weights do not match the input schema")
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        weights_path = output_directory / WEIGHTS_FILENAME
        with weights_path.open("wb") as destination:
            np.save(
                destination,
                np.append(model.weights, np.float32(model.bias)),
                allow_pickle=False,
            )
        onnx_path = output_directory / ONNX_FILENAME
        _export_onnx(model, input_schema, onnx_path)
        metadata = ModelArtifactMetadata(
            model_name="linear_pixel_logistic",
            model_version=model_version,
            training_dataset_version=training_dataset_version,
            runtime="onnx",
            input_schema=input_schema,
            normalization=model.normalization,
            class_mapping={0: "background", 1: "building"},
            checksum=file_content_hash(weights_path),
            onnx_checksum=file_content_hash(onnx_path),
            evaluation_metrics=evaluation_metrics,
        )
        (output_directory / METADATA_FILENAME).write_text(
            json.dumps(metadata.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        return metadata
    except OSError as exc:
        raise StorageError(f"failed to save segmentation artifact: {output_directory}") from exc


class SegmentationInference:
    def __init__(self, metadata: ModelArtifactMetadata, session: ort.InferenceSession) -> None:
        self.metadata = metadata
        self._session = session

    def predict_probabilities(
        self, image: NDArray[np.uint8], channel_order: tuple[str, ...]
    ) -> NDArray[np.float32]:
        schema = self.metadata.input_schema
        if image.dtype != np.uint8:
            raise InferenceValidationError("input dtype must be uint8")
        if image.shape != (schema.height, schema.width, schema.channels):
            raise InferenceValidationError("input shape does not match the artifact schema")
        if channel_order != schema.channel_order:
            raise InferenceValidationError("input channel order does not match the artifact schema")
        batch = image[np.newaxis].astype(np.float32) / 255.0
        output = self._session.run(["probabilities"], {"image": batch})[0]
        return cast(NDArray[np.float32], np.asarray(output[0, :, :, 0], dtype=np.float32))

    def predict(
        self,
        image: NDArray[np.uint8],
        channel_order: tuple[str, ...],
        threshold: float = 0.5,
    ) -> NDArray[np.uint8]:
        if not 0 < threshold < 1:
            raise InferenceValidationError("threshold must be between zero and one")
        return cast(
            NDArray[np.uint8],
            (self.predict_probabilities(image, channel_order) >= threshold).astype(np.uint8),
        )


def load_model_artifact(output_directory: Path) -> SegmentationInference:
    """Validate every artifact file and load only the declared ONNX runtime."""

    metadata_path = output_directory / METADATA_FILENAME
    weights_path = output_directory / WEIGHTS_FILENAME
    onnx_path = output_directory / ONNX_FILENAME
    try:
        metadata = ModelArtifactMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise ArtifactValidationError("model artifact metadata is missing or invalid") from exc
    if metadata.runtime != "onnx":
        raise ArtifactValidationError(f"unsupported declared runtime: {metadata.runtime}")
    try:
        if file_content_hash(weights_path) != metadata.checksum:
            raise ArtifactValidationError("model weights checksum mismatch")
        if file_content_hash(onnx_path) != metadata.onnx_checksum:
            raise ArtifactValidationError("ONNX checksum mismatch")
        with weights_path.open("rb") as source:
            parameters = np.load(source, allow_pickle=False)
        if (
            parameters.shape != (metadata.input_schema.channels + 1,)
            or not np.isfinite(parameters).all()
        ):
            raise ArtifactValidationError("model weights shape or values are invalid")
        if len(metadata.normalization.mean) != metadata.input_schema.channels:
            raise ArtifactValidationError("normalization does not match input channels")
        session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
        input_metadata = session.get_inputs()[0]
        if input_metadata.name != "image" or input_metadata.shape[1:] != [
            metadata.input_schema.height,
            metadata.input_schema.width,
            metadata.input_schema.channels,
        ]:
            raise ArtifactValidationError("ONNX input schema is incompatible")
        output_metadata = session.get_outputs()[0]
        if output_metadata.name != "probabilities" or output_metadata.shape[1:] != [
            metadata.input_schema.height,
            metadata.input_schema.width,
            1,
        ]:
            raise ArtifactValidationError("ONNX output schema is incompatible")
    except ArtifactValidationError:
        raise
    except (OSError, ValueError, Fail, InvalidArgument, RuntimeException) as exc:
        raise ArtifactValidationError("failed to load declared ONNX artifact") from exc
    return SegmentationInference(metadata, session)
