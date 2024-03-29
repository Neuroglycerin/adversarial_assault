"""Tool which runs all attacks against all defenses and computes results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import os
import sys
import subprocess
import numpy as np
from PIL import Image
import hashlib
from checksumdir import dirhash
import errno
import time


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description='Tool to run attacks and defenses.')
  parser.add_argument('--attacks_dir', required=True,
                      help='Location of all attacks.')
  parser.add_argument('--targeted_attacks_dir', required=True,
                      help='Location of all targeted attacks.')
  parser.add_argument('--dataset_dir', required=True,
                      help='Location of the dataset.')
  parser.add_argument('--dataset_metadata', required=True,
                      help='Location of the dataset metadata.')
  parser.add_argument('--intermediate_results_dir', required=True,
                      help='Directory to store intermediate results.')
  parser.add_argument('--output_dir', required=True,
                      help=('Output directory.'))
  parser.add_argument('--epsilon', required=False, type=int, default=16,
                      help='Maximum allowed size of adversarial perturbation')
  parser.add_argument('--gpu', dest='use_gpu', action='store_true')
  parser.add_argument('--nogpu', dest='use_gpu', action='store_false')
  parser.set_defaults(use_gpu=False)
  return parser.parse_args()


class Submission(object):
  """Base class for all submissions."""

  def __init__(self, directory, container, entry_point, use_gpu):
    """Initializes instance of Submission class.

    Args:
      directory: location of the submission.
      container: URL of Docker container which should be used to run submission.
      entry_point: entry point script, which invokes submission.
      use_gpu: whether to use Docker with GPU or not.
    """
    self.name = os.path.basename(directory)
    self.directory = directory
    self.container = container
    self.entry_point = entry_point
    self.use_gpu = use_gpu
    self.sec_per_100_samples = None
    self.output_count = 0

  def docker_binary(self):
    """Returns appropriate Docker binary to use."""
    return 'nvidia-docker' if self.use_gpu else 'docker'

  def __eq__(self, other):
    return self.name == other.name

  def __ne__(self, other):
    return not (self == other)

  def __lt__(self, other):
    return self.name < other.name

  def __le__(self, other):
    return self.name <= other.name

  def __gt__(self, other):
    return self.name > other.name

  def __ge__(self, other):
    return self.name >= other.name


class Attack(Submission):
  """Class which stores and runs attack."""

  def __init__(self, directory, container, entry_point, use_gpu):
    """Initializes instance of Attack class."""
    super(Attack, self).__init__(directory, container, entry_point, use_gpu)

  def run(self, input_dir, output_dir, epsilon):
    """Runs attack inside Docker.

    Args:
      input_dir: directory with input (dataset).
      output_dir: directory where output (adversarial images) should be written.
      epsilon: maximum allowed size of adversarial perturbation,
        should be in range [0, 255].
    """
    print('Running attack ', self.name)
    sys.stdout.flush()
    t0 = time.time()
    cmd = [self.docker_binary(), 'run',
           '-v', '{0}:/input_images'.format(input_dir),
           '-v', '{0}:/output_images'.format(output_dir),
           '-v', '{0}:/code'.format(self.directory),
           '-w', '/code',
           self.container,
           './' + self.entry_point,
           '/input_images',
           '/output_images',
           str(epsilon),
           '2>&1 | tee -a {0}/stdout.log'.format(output_dir)]
    print(' '.join(cmd))
    subprocess.call(cmd)
    t1 = time.time()
    duration = t1-t0
    n_files = len([name for name in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, name))])
    print('Attack {} took {} seconds and outputed {} images'.format(
        self.name, duration, n_files))
    self.output_count = n_files
    if n_files == 0:
        self.sec_per_100_samples = None
    else:
        self.sec_per_100_samples = 100 * duration / n_files
    filepath = os.path.join(output_dir, 'time_per_100.txt')
    with open(filepath, 'w') as f:
      f.write(str(self.sec_per_100_samples))

  def maybe_run(self, hash_folder, input_dir, output_dir, epsilon):
    # Check whether we already computed images for this *exact* submission
    # The file name encodes the submission's name and the target dir
    fname = '{}_{}'.format(
      self.name,
      hashlib.sha1(output_dir).hexdigest(),
    )
    # We encode the current request with the hash of the submission, and
    # the input data (which consists of the input directory and eps)
    # We assume that the input directory's contents are static.
    expected_hash = '{}_{}_{}'.format(
      dirhash(self.directory, 'sha1'),
      hashlib.sha1(input_dir).hexdigest(),
      epsilon,
    )
    filepath = os.path.join(hash_folder, 'attacks', fname)
    if not os.path.isfile(filepath):
      pass
    else:
      with open(filepath, 'r') as f:
        last_hash = f.read()
      if last_hash == expected_hash:
        print('Using cached output for ' + self.directory)
        return
      else:
        os.remove(filepath)
    # We do need to run the code and generate these attacks
    self.run(input_dir, output_dir, epsilon)
    # Remember that this is what we last output
    with open(filepath, 'w') as f:
      f.write(expected_hash)


def count_lines_in_file(fname):
    i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def read_submissions_from_directory(dirname, use_gpu):
  """Scans directory and read all submissions.

  Args:
    dirname: directory to scan.
    use_gpu: whether submissions should use GPU. This argument is
      used to pick proper Docker container for each submission and create
      instance of Attack or Defense class.

  Returns:
    List with submissions (subclasses of Submission class).
  """
  result = []
  for sub_dir in os.listdir(dirname):
    submission_path = os.path.join(dirname, sub_dir)
    try:
      if not os.path.isdir(submission_path):
        continue
      if not os.path.exists(os.path.join(submission_path, 'metadata.json')):
        continue
      with open(os.path.join(submission_path, 'metadata.json')) as f:
        metadata = json.load(f)
      if use_gpu and ('container_gpu' in metadata):
        container = metadata['container_gpu']
      else:
        container = metadata['container']
      entry_point = metadata['entry_point']
      submission_type = metadata['type']
      if submission_type == 'attack' or submission_type == 'targeted_attack':
        submission = Attack(submission_path, container, entry_point, use_gpu)
      elif submission_type == 'defense':
        submission = Defense(submission_path, container, entry_point, use_gpu)
      else:
        raise ValueError('Invalid type of submission: %s', submission_type)
      result.append(submission)
    except (IOError, KeyError, ValueError):
      print('Failed to read submission from directory ', submission_path)
  return result


class AttacksOutput(object):
  """Helper class to store data about images generated by attacks."""

  def __init__(self,
               dataset_dir,
               attacks_output_dir,
               targeted_attacks_output_dir,
               all_adv_examples_dir,
               epsilon):
    """Initializes instance of AttacksOutput class.

    Args:
      dataset_dir: location of the dataset.
      attacks_output_dir: where to write results of attacks.
      targeted_attacks_output_dir: where to write results of targeted attacks.
      all_adv_examples_dir: directory to copy all adversarial examples from
        all attacks.
      epsilon: maximum allowed size of adversarial perturbation.
    """
    self.attacks_output_dir = attacks_output_dir
    self.targeted_attacks_output_dir = targeted_attacks_output_dir
    self.all_adv_examples_dir = all_adv_examples_dir
    self._load_dataset_clipping(dataset_dir, epsilon)
    self._output_image_idx = 0
    self._output_to_attack_mapping = {}
    self._attack_image_count = 0
    self._targeted_attack_image_count = 0
    self._attack_names = set()
    self._targeted_attack_names = set()
    self.sec_per_100_samples_attack = {}
    self.sec_per_100_samples_targeted_attack = {}

  def _load_dataset_clipping(self, dataset_dir, epsilon):
    """Helper method which loads dataset and determines clipping range.

    Args:
      dataset_dir: location of the dataset.
      epsilon: maximum allowed size of adversarial perturbation.
    """
    self.dataset_max_clip = {}
    self.dataset_min_clip = {}
    self._dataset_image_count = 0
    for fname in os.listdir(dataset_dir):
      if not fname.endswith('.png'):
        continue
      image_id = fname[:-4]
      image = np.array(
          Image.open(os.path.join(dataset_dir, fname)).convert('RGB'))
      image = image.astype('int32')
      self._dataset_image_count += 1
      self.dataset_max_clip[image_id] = np.clip(image + epsilon,
                                                0,
                                                255).astype('uint8')
      self.dataset_min_clip[image_id] = np.clip(image - epsilon,
                                                0,
                                                255).astype('uint8')

  def clip_and_copy_attack_outputs(self, attack_name, is_targeted):
    """Clips results of attack and copy it to directory with all images.

    Args:
      attack_name: name of the attack.
      is_targeted: if True then attack is targeted, otherwise non-targeted.
    """
    if is_targeted:
      self._targeted_attack_names.add(attack_name)
    else:
      self._attack_names.add(attack_name)
    attack_dir = os.path.join(self.targeted_attacks_output_dir
                              if is_targeted
                              else self.attacks_output_dir,
                              attack_name)

    dur = load_duration(os.path.join(attack_dir, 'time_per_100.txt'))
    if is_targeted:
      self.sec_per_100_samples_targeted_attack[attack_name] = dur
    else:
      self.sec_per_100_samples_attack[attack_name] = dur

    for fname in os.listdir(attack_dir):
      if not (fname.endswith('.png') or fname.endswith('.jpg')):
        continue
      image_id = fname[:-4]
      if image_id not in self.dataset_max_clip:
        continue
      image_max_clip = self.dataset_max_clip[image_id]
      image_min_clip = self.dataset_min_clip[image_id]
      adversarial_image = np.array(
          Image.open(os.path.join(attack_dir, fname)).convert('RGB'))
      clipped_adv_image = np.clip(adversarial_image,
                                  image_min_clip,
                                  image_max_clip)
      output_basename = '{0:08d}'.format(self._output_image_idx)
      self._output_image_idx += 1
      self._output_to_attack_mapping[output_basename] = (attack_name,
                                                         is_targeted,
                                                         image_id)
      if is_targeted:
        self._targeted_attack_image_count += 1
      else:
        self._attack_image_count += 1
      Image.fromarray(clipped_adv_image).save(
          os.path.join(self.all_adv_examples_dir, output_basename + '.png'))

  @property
  def attack_names(self):
    """Returns list of all non-targeted attacks."""
    return self._attack_names

  @property
  def targeted_attack_names(self):
    """Returns list of all targeted attacks."""
    return self._targeted_attack_names

  @property
  def attack_image_count(self):
    """Returns number of all images generated by non-targeted attacks."""
    return self._attack_image_count

  @property
  def dataset_image_count(self):
    """Returns number of all images in the dataset."""
    return self._dataset_image_count

  @property
  def targeted_attack_image_count(self):
    """Returns number of all images generated by targeted attacks."""
    return self._targeted_attack_image_count

  def image_by_base_filename(self, filename):
    """Returns information about image based on it's filename."""
    return self._output_to_attack_mapping[filename]


class DatasetMetadata(object):
  """Helper class which loads and stores dataset metadata."""

  def __init__(self, filename):
    """Initializes instance of DatasetMetadata."""
    self._true_labels = {}
    self._target_classes = {}
    with open(filename) as f:
      reader = csv.reader(f)
      header_row = next(reader)
      try:
        row_idx_image_id = header_row.index('ImageId')
        row_idx_true_label = header_row.index('TrueLabel')
        row_idx_target_class = header_row.index('TargetClass')
      except ValueError:
        raise IOError('Invalid format of dataset metadata.')
      for row in reader:
        if len(row) < len(header_row):
          # skip partial or empty lines
          continue
        try:
          image_id = row[row_idx_image_id]
          self._true_labels[image_id] = int(row[row_idx_true_label])
          self._target_classes[image_id] = int(row[row_idx_target_class])
        except (IndexError, ValueError):
          raise IOError('Invalid format of dataset metadata')

  def get_true_label(self, image_id):
    """Returns true label for image with given ID."""
    return self._true_labels[image_id]

  def get_target_class(self, image_id):
    """Returns target class for image with given ID."""
    return self._target_classes[image_id]

  def save_target_classes(self, filename):
    """Saves target classed for all dataset images into given file."""
    with open(filename, 'w') as f:
      for k, v in self._target_classes.items():
        f.write('{0}.png,{1}\n'.format(k, v))


def load_defense_output(filename):
  """Loads output of defense from given file."""
  result = {}
  with open(filename) as f:
    for row in csv.reader(f):
      try:
        image_filename = row[0]
        if image_filename.endswith('.png') or image_filename.endswith('.jpg'):
          image_filename = image_filename[:image_filename.rfind('.')]
        label = int(row[1])
      except (IndexError, ValueError):
        continue
      result[image_filename] = label
  return result


def load_duration(filename):
  try:
    with open(filename) as f:
      return float(f.read())
  except ValueError:
    return None


def compute_and_save_scores_and_ranking(attacks_output,
                                        dataset_meta,
                                        output_dir):
  """Computes scores and ranking and saves it.

  Args:
    attacks_output: output of attacks, instance of AttacksOutput class.
    defenses_output: outputs of defenses. Dictionary of dictionaries, key in
      outer dictionary is name of the defense, key of inner dictionary is
      name of the image, value of inner dictionary is classification label.
    dataset_meta: dataset metadata, instance of DatasetMetadata class.
    output_dir: output directory where results will be saved.
    save_all_classification: If True then classification results of all
      defenses on all images produces by all attacks will be saved into
      all_classification.csv file. Useful for debugging.

  This function saves following files into output directory:
    accuracy_on_attacks.csv: matrix with number of correctly classified images
      for each pair of defense and attack.
    accuracy_on_targeted_attacks.csv: matrix with number of correctly classified
      images for each pair of defense and targeted attack.
    hit_target_class.csv: matrix with number of times defense classified image
      as specified target class for each pair of defense and targeted attack.
    defense_ranking.csv: ranking and scores of all defenses.
    attack_ranking.csv: ranking and scores of all attacks.
    targeted_attack_ranking.csv: ranking and scores of all targeted attacks.
    all_classification.csv: results of classification of all defenses on
      all images produced by all attacks. Only saved if save_all_classification
      argument is True.
  """
  def write_ranking(filename, header, names, scores):
    """Helper method which saves submissions' scores and names."""
    order = np.argsort(scores)[::-1]
    with open(filename, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(header)
      for idx in order:
        writer.writerow([names[idx], scores[idx]])

  def write_score_matrix(filename, scores, row_names, column_names):
    """Helper method which saves score matrix."""
    result = np.pad(scores, ((1, 0), (1, 0)), 'constant').astype(np.object)
    result[0, 0] = ''
    result[1:, 0] = row_names
    result[0, 1:] = column_names
    np.savetxt(filename, result, fmt='%s', delimiter=',')

  attack_names = sorted(list(attacks_output.attack_names))
  attack_names_idx = {name: index for index, name in enumerate(attack_names)}
  targeted_attack_names = sorted(list(attacks_output.targeted_attack_names))
  targeted_attack_names_idx = {name: index
                               for index, name
                               in enumerate(targeted_attack_names)}

  # Save matrices.
  attacks_duration = []
  for name in attack_names:
    attacks_duration.append(
      attacks_output.sec_per_100_samples_attack[name])
  targeted_attacks_duration = []
  for name in targeted_attack_names:
    targeted_attacks_duration.append(
      attacks_output.sec_per_100_samples_targeted_attack[name])
  write_ranking(
      os.path.join(output_dir, 'duration_attack.csv'),
      ['AttackName', 'DurationFor100Samples'], attack_names,
      attacks_duration)
  write_ranking(
      os.path.join(output_dir, 'duration_targeted_attack.csv'),
      ['AttackName', 'DurationFor100Samples'], targeted_attack_names,
      targeted_attacks_duration)


def main():
  args = parse_args()
  hash_dir = os.path.join(args.intermediate_results_dir, 'hashes')
  attacks_output_dir = os.path.join(args.intermediate_results_dir,
                                    'attacks_output')
  targeted_attacks_output_dir = os.path.join(args.intermediate_results_dir,
                                             'targeted_attacks_output')
  all_adv_examples_dir = os.path.join(args.intermediate_results_dir,
                                      'all_adv_examples')

  # Load dataset metadata.
  dataset_meta = DatasetMetadata(args.dataset_metadata)

  # Load attacks and defenses.
  attacks = [
      a for a in read_submissions_from_directory(args.attacks_dir,
                                                 args.use_gpu)
      if isinstance(a, Attack)
  ]
  targeted_attacks = [
      a for a in read_submissions_from_directory(args.targeted_attacks_dir,
                                                 args.use_gpu)
      if isinstance(a, Attack)
  ]
  attacks = sorted(attacks)
  targeted_attacks = sorted(targeted_attacks)

  print('Found attacks: ', [a.name for a in attacks])
  print('Found tageted attacks: ', [a.name for a in targeted_attacks])

  def maybe_make_dir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(dirname):
            pass
        else:
            raise

  # Prepare subdirectories for intermediate results.
  maybe_make_dir(hash_dir)
  maybe_make_dir(os.path.join(hash_dir, 'attacks'))
  maybe_make_dir(attacks_output_dir)
  maybe_make_dir(targeted_attacks_output_dir)
  maybe_make_dir(all_adv_examples_dir)
  for a in attacks:
    maybe_make_dir(os.path.join(attacks_output_dir, a.name))
  for a in targeted_attacks:
    maybe_make_dir(os.path.join(targeted_attacks_output_dir, a.name))

  # Run all non-targeted attacks.
  attacks_output = AttacksOutput(args.dataset_dir,
                                 attacks_output_dir,
                                 targeted_attacks_output_dir,
                                 all_adv_examples_dir,
                                 args.epsilon)
  for a in attacks:
    a.maybe_run(hash_dir,
          args.dataset_dir,
          os.path.join(attacks_output_dir, a.name),
          args.epsilon)
    attacks_output.clip_and_copy_attack_outputs(a.name, False)

  # Run all targeted attacks.
  dataset_meta.save_target_classes(os.path.join(args.dataset_dir,
                                                'target_class.csv'))
  for a in targeted_attacks:
    a.maybe_run(hash_dir,
          args.dataset_dir,
          os.path.join(targeted_attacks_output_dir, a.name),
          args.epsilon)
    attacks_output.clip_and_copy_attack_outputs(a.name, True)

  # Compute and save scoring.
  compute_and_save_scores_and_ranking(attacks_output,
                                      dataset_meta,
                                      args.output_dir)


if __name__ == '__main__':
  main()
