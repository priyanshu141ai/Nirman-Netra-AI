"""Phase 8 integrated municipal processing schema."""

from alembic import op

from nirman_netra.persistence.schema import metadata

revision = "0001_phase8"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    metadata.create_all(bind=op.get_bind(), checkfirst=False)


def downgrade() -> None:
    metadata.drop_all(bind=op.get_bind(), checkfirst=False)
