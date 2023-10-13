"""nerf_dataset dataset."""

import json
import numpy as np
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for nerf_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def __init__(self, path, *args, **kwargs):
    self.path = path
    super(Builder).__init__(*args, **kwargs)

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(None, None, 3)),
            'c2w': tfds.features.Tensor(shape=(3,4)), # R|t
        }),
        supervised_keys=('image', 'c2w'),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {
        'train': self._generate_examples(self.path, 'transforms_train.json'),
        'test': self._generate_examples(self.path, 'transforms_test.json'),
        'val': self._generate_examples(self.path, 'transforms_val.json'),
    }

  def _generate_examples(self, scene_path, transforms_file):
    """Yields examples."""
    with open(scene_path/transforms_file, 'r') as f:
      data = json.load(f)

      for i, frame in enumerate(data["frames"]):
        img_path = scene_path / f"{frame['file_path']}.png"
        transform = np.array(frame['transform_matrix'])
        yield i, {
          'image': img_path,
          'c2w': transform[:3, :],
        }
