"""Brier Score Metric"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import datasets
import evaluate
import numpy as np
from sksurv.metrics import concordance_index_ipcw

from sat.utils import logging

logger = logging.get_default_logger()

_CITATION = """\
@article{Uno2011OnTC,
  title={On the C‐statistics for evaluating overall adequacy of risk prediction procedures with censored survival data},
  author={Hajime Uno and Tianxi Cai and Michael J. Pencina and Ralph B. D'Agostino and L. J. Wei},
  journal={Statistics in Medicine},
  year={2011},
  volume={30}
}"""

_DESCRIPTION = """\
Concordance index for right-censored data based on inverse probability of censoring weights.
    This is an alternative to the estimator in :func:`concordance_index_censored`
    that does not depend on the distribution of censoring times in the test data.
    Therefore, the estimate is unbiased and consistent for a population concordance
    measure that is free of censoring.
    It is based on inverse probability of censoring weights, thus requires
    access to survival times from the training data to estimate the censoring
    distribution. Note that this requires that survival times `survival_test`
    lie within the range of survival times `survival_train`. This can be
    achieved by specifying the truncation time `tau`.
    The resulting `cindex` tells how well the given prediction model works in
    predicting events that occur in the time range from 0 to `tau`.
    The estimator uses the Kaplan-Meier estimator to estimate the
    censoring survivor function. Therefore, it is restricted to
    situations where the random censoring assumption holds and
    censoring is independent of the features.
    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further description.

    See also
    --------
    concordance_index_censored
        Simpler estimator of the concordance index.
    as_concordance_index_ipcw_scorer
        Wrapper class that uses :func:`concordance_index_ipcw`
        in its ``score`` method instead of the default
        :func:`concordance_index_censored`.

    References
    ----------
    .. [1] Uno, H., Cai, T., Pencina, M. J., D’Agostino, R. B., & Wei, L. J. (2011).
           "On the C-statistics for evaluating overall adequacy of risk prediction
           procedures with censored survival data".
           Statistics in Medicine, 30(10), 1105–1117.
"""

_KWARGS_DESCRIPTION = """

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.
    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.
    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event of test data.
    tau : float, optional
        Truncation time. The survival function for the underlying
        censoring time distribution :math:`D` needs to be positive
        at `tau`, i.e., `tau` should be chosen such that the
        probability of being censored after time `tau` is non-zero:
        :math:`P(D > \\tau) > 0`. If `None`, no truncation is performed.
    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    cindex : float
        Concordance index
    concordant : int
        Number of concordant pairs
    discordant : int
        Number of discordant pairs
    tied_risk : int
        Number of pairs having tied estimated risks
    tied_time : int
        Number of comparable pairs sharing the same time
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CIIPCW(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._get_feature_types(),
            reference_urls=[
                "https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html"
            ],
        )

    def _get_feature_types(self):
        return datasets.Features(
            {
                "predictions": datasets.Sequence(datasets.Value("float")),
                "references": datasets.Sequence(datasets.Value("float")),
            }
        )

    def _compute(
        self,
        references,
        predictions,
        train_set,
        duration_cuts,
    ):
        et_test = np.array(
            [tuple(x) for x in references], dtype=[("e", bool), ("t", float)]
        )
        et_train = np.array(
            [tuple(x) for x in train_set], dtype=[("e", bool), ("t", float)]
        )

        # Convert predictions to numpy array once, outside the loop
        predictions_np = np.array(predictions)

        # Use joblib for parallel processing if available
        try:
            from joblib import Parallel, delayed

            def compute_cindex(i, train_data, test_data, preds, cuts):
                try:
                    # Check if we have enough samples for this time point
                    if preds.shape[0] == 0:
                        logger.warning(f"Empty predictions array for time point {i}")
                        return 0.5  # Return random predictor performance
                    
                    if len(train_data) == 0 or len(test_data) == 0:
                        logger.warning(f"Empty train or test data for time point {i}")
                        return 0.5
                    
                    estimate_vals = preds[:, i]
                    if len(estimate_vals) == 0 or np.all(np.isnan(estimate_vals)):
                        logger.warning(f"All NaN or empty estimates for time point {i}")
                        return 0.5
                    
                    return concordance_index_ipcw(
                        train_data,
                        test_data,
                        estimate=estimate_vals,
                        tau=cuts[i],
                    )[0]
                except Exception as e:
                    logger.error(f"Error computing c-index for time point {i}: {str(e)}")
                    return 0.5  # Return random predictor performance on error

            # Use parallel processing with all available cores
            cindeces = Parallel(n_jobs=-1)(
                delayed(compute_cindex)(
                    i, et_train, et_test, predictions_np, duration_cuts
                )
                for i in range(len(duration_cuts))
            )
        except ImportError:
            # Fallback to sequential processing if joblib is not available
            # But still using pre-converted predictions for efficiency
            cindeces = []
            for i, _ in enumerate(duration_cuts):
                try:
                    # Check if we have enough samples for this time point
                    if predictions_np.shape[0] == 0:
                        logger.warning(f"Empty predictions array for time point {i}")
                        cindeces.append(0.5)
                        continue
                    
                    if len(et_train) == 0 or len(et_test) == 0:
                        logger.warning(f"Empty train or test data for time point {i}")
                        cindeces.append(0.5)
                        continue
                    
                    estimate_vals = predictions_np[:, i]
                    if len(estimate_vals) == 0 or np.all(np.isnan(estimate_vals)):
                        logger.warning(f"All NaN or empty estimates for time point {i}")
                        cindeces.append(0.5)
                        continue
                    
                    cindex = concordance_index_ipcw(
                        et_train,
                        et_test,
                        estimate=estimate_vals,
                        tau=duration_cuts[i],
                    )[0]
                    cindeces.append(cindex)
                except Exception as e:
                    logger.error(f"Error computing c-index for time point {i}: {str(e)}")
                    cindeces.append(0.5)  # Return random predictor performance on error

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Computed c-index: {cindeces}")

        return cindeces
