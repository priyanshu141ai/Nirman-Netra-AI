"""Persist complete integration records and correct evidence lineage."""

from alembic import op

revision = "0002_durable_repository"
down_revision = "0001_phase8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE municipalities ADD COLUMN IF NOT EXISTS payload JSON")
    op.execute("ALTER TABLE image_pairs ADD COLUMN IF NOT EXISTS payload JSON")
    op.execute(
        "ALTER TABLE evidence_assets "
        "DROP CONSTRAINT IF EXISTS evidence_assets_source_asset_id_fkey"
    )
    op.create_foreign_key(
        "evidence_assets_source_asset_id_fkey",
        "evidence_assets",
        "evidence_assets",
        ["source_asset_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "evidence_assets_source_asset_id_fkey", "evidence_assets", type_="foreignkey"
    )
    op.create_foreign_key(
        "evidence_assets_source_asset_id_fkey",
        "evidence_assets",
        "imagery_assets",
        ["source_asset_id"],
        ["id"],
    )
    op.drop_column("image_pairs", "payload")
    op.drop_column("municipalities", "payload")
