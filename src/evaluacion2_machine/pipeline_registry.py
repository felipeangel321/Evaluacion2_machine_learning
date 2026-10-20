from __future__ import annotations

from kedro.pipeline import Pipeline

from evaluacion2_machine.pipelines.data_cleaning import create_pipeline as cleaning_pipeline
from evaluacion2_machine.pipelines.data_transformation import create_pipeline as transformation_pipeline
from evaluacion2_machine.pipelines.modeling import create_pipeline as modeling_pipeline
from evaluacion2_machine.pipelines.hyperparameter_tuning import create_pipeline as hyperparameter_pipeline
from evaluacion2_machine.pipelines.evaluation import create_pipeline as evaluation_pipeline
from evaluacion2_machine.pipelines.unsupervised import create_pipeline as unsupervised_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    dp = cleaning_pipeline() + transformation_pipeline()
    return {
        "data_preprocessing": dp,
        "data_cleaning": cleaning_pipeline(),
        "data_transformation": transformation_pipeline(),
        "modeling": modeling_pipeline(),
        "hyperparameter_tuning": hyperparameter_pipeline(),
        "evaluation": evaluation_pipeline(),
        "unsupervised": unsupervised_pipeline(),
        "__default__": (
            dp
            + modeling_pipeline()
            + hyperparameter_pipeline()
            + evaluation_pipeline()
            + unsupervised_pipeline()
        ),
    }
