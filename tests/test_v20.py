"""Safety checks for the new isolated v20 runner; no real data or model fitting."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('gmm_v20',Path(__file__).resolve().parents[1]/'scripts/gmm_v20.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

class TestV20(unittest.TestCase):
    def test_output_cannot_replace_or_contain_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=root/'data';data.mkdir()
            for target in [data,data/'new',root]:
                with self.assertRaises(ValueError):m.output_path(target,[data])
            self.assertEqual(m.output_path(root/'run',[data]),(root/'run').resolve())
    def test_output_cannot_follow_symlink_into_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=root/'data';data.mkdir();alias=root/'alias';alias.symlink_to(data,target_is_directory=True)
            with self.assertRaises(ValueError):m.output_path(alias/'new',[data])
    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'run';out.mkdir()
            with self.assertRaises(ValueError):m.output_path(out,[root/'data'])
    def test_tampered_or_traversing_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);f=root/'a';f.write_text('original')
            self.assertEqual(m.verify_manifest(root,{'a':m.sha256(f)}),1)
            with self.assertRaises(ValueError):m.verify_manifest(root,{'a':'wrong'})
            with self.assertRaises(ValueError):m.verify_manifest(root,{'../a':'wrong'})
    def test_distributed_update_is_intact(self):
        self.assertGreater(m.verify_update(),0)

if __name__=='__main__':unittest.main()
