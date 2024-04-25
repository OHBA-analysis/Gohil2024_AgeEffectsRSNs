"""Dipole sign flipping.

"""

from glob import glob
from dask.distributed import Client

from osl import utils
from osl.source_recon import find_template_subject, run_src_batch

SRC_DIR = "data/src"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Subjects to sign flip
    subjects = []
    for path in sorted(glob(SRC_DIR + "/*/rhino/parc-raw.fif")):
        subject = path.split("/")[-3]
        subjects.append(subject)

    # Find a good template subject to align other subjects to
    template = find_template_subject(SRC_DIR, subjects, n_embeddings=15, standardize=True)

    # Settings for batch processing
    config = f"""
        source_recon:
        - fix_sign_ambiguity:
            template: {template}
            n_embeddings: 15
            standardize: True
            n_init: 5
            n_iter: 5000
            max_flips: 20
    """

    # Setup parallel processing
    client = Client(n_workers=16, threads_per_worker=1)

    # Do the sign flipping
    run_src_batch(config, SRC_DIR, subjects, dask_client=True)
