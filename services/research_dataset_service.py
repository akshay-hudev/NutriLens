from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkDatasetConfig:
    name: str
    root: Path
    images_dir: Path
    masks_dir: Path
    depth_dir: Path
    meshes_dir: Path
    nutrition_labels_path: Path


class ResearchDatasetService:
    """Configuration hook for future MetaFood3D-style evaluation datasets."""

    def __init__(self, name: str, root: Path | None) -> None:
        self.name = name
        self.root = Path(root) if root else None

    @property
    def available(self) -> bool:
        return self.root is not None and self.root.exists()

    def config(self) -> BenchmarkDatasetConfig:
        root = self.root or Path("<NUTRILENS_BENCHMARK_DATA_ROOT>")
        return BenchmarkDatasetConfig(
            name=self.name,
            root=root,
            images_dir=root / "images",
            masks_dir=root / "masks",
            depth_dir=root / "depth",
            meshes_dir=root / "meshes",
            nutrition_labels_path=root / "nutrition_labels.csv",
        )

    def describe(self) -> dict:
        cfg = self.config()
        return {
            "name": cfg.name,
            "configured": self.available,
            "expected_layout": {
                "images": str(cfg.images_dir),
                "masks": str(cfg.masks_dir),
                "depth_maps": str(cfg.depth_dir),
                "meshes": str(cfg.meshes_dir),
                "nutrition_labels": str(cfg.nutrition_labels_path),
            },
        }
