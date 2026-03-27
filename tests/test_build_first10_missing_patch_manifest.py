from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bolt.generate.scripts.build_first10_missing_patch_manifest import build_manifest


class BuildFirst10MissingPatchManifestTests(unittest.TestCase):
    def test_build_manifest_uses_allowlist_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowlist_path = root / "allowlist.json"
            metadata_dir = root / "metadata"
            mask_dir = root / "masks"
            image_dir = root / "images"
            xml_dir = root / "xml"
            for path in (metadata_dir, mask_dir, image_dir, xml_dir):
                path.mkdir(parents=True, exist_ok=True)

            allowlist_path.write_text(
                json.dumps({"records": [{"asset_id": "sample-001", "xcf_path": "x.xcf"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            (metadata_dir / "sample-001.json").write_text(
                json.dumps({"image_name": "sample.jpg"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (mask_dir / "sample-001_mask.png").write_bytes(b"mask")
            (image_dir / "sample.jpg").write_bytes(b"img")
            (xml_dir / "sample.xml").write_text(
                "<annotation><object><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox></object></annotation>",
                encoding="utf-8",
            )

            summary = build_manifest(
                allowlist_path=allowlist_path,
                metadata_dir=metadata_dir,
                mask_dir=mask_dir,
                image_dir=image_dir,
                xml_dir=xml_dir,
            )

            self.assertEqual(summary["record_count"], 1)
            record = summary["records"][0]
            self.assertEqual(record["asset_id"], "sample-001")
            self.assertEqual(record["donor_box"], [1, 2, 3, 4])

    def test_build_manifest_falls_back_to_asset_id_image_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowlist_path = root / "allowlist.json"
            metadata_dir = root / "metadata"
            mask_dir = root / "masks"
            image_dir = root / "images"
            xml_dir = root / "xml"
            for path in (metadata_dir, mask_dir, image_dir, xml_dir):
                path.mkdir(parents=True, exist_ok=True)

            allowlist_path.write_text(
                json.dumps({"records": [{"asset_id": "sample-001", "xcf_path": "x.xcf"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            (metadata_dir / "sample-001.json").write_text(
                json.dumps({"image_name": "legacy.jpg"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (mask_dir / "sample-001_mask.png").write_bytes(b"mask")
            (image_dir / "sample-001.jpg").write_bytes(b"img")
            (xml_dir / "legacy.xml").write_text(
                "<annotation><object><bndbox><xmin>5</xmin><ymin>6</ymin><xmax>7</xmax><ymax>8</ymax></bndbox></object></annotation>",
                encoding="utf-8",
            )

            summary = build_manifest(
                allowlist_path=allowlist_path,
                metadata_dir=metadata_dir,
                mask_dir=mask_dir,
                image_dir=image_dir,
                xml_dir=xml_dir,
            )

            record = summary["records"][0]
            self.assertEqual(Path(record["image_path"]).name, "sample-001.jpg")
            self.assertEqual(record["donor_box"], [5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
