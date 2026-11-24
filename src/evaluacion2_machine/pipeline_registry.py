from __future__ import annotations

from kedro.pipeline import Pipeline

from evaluacion2_machine.pipelines.data_cleaning import create_pipeline as cleaning_pipeline
from evaluacion2_machine.pipelines.data_transformation import create_pipeline as transformation_pipeline
from evaluacion2_machine.pipelines.modeling import create_pipeline as modeling_pipeline
from evaluacion2_machine.pipelines.hyperparameter_tuning import create_pipeline as hyperparameter_pipeline
from evaluacion2_machine.pipelines.ensembles import create_pipeline as ensembles_pipeline
from evaluacion2_machine.pipelines.evaluation import create_pipeline as evaluation_pipeline
from evaluacion2_machine.pipelines.unsupervised import create_pipeline as unsupervised_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    dp = cleaning_pipeline() + transformation_pipeline()
    # modeling entrega los modelos iniciales en memoria a hyperparameter_tuning,
    # por eso ambos se ejecutan juntos en "training"
    training = modeling_pipeline() + hyperparameter_pipeline()
    supervised = training + ensembles_pipeline() + evaluation_pipeline()
    return {
        "data_preprocessing": dp,
        "data_cleaning": cleaning_pipeline(),
        "data_transformation": transformation_pipeline(),
        "training": training,
        "ensembles": ensembles_pipeline(),
        "evaluation": evaluation_pipeline(),
        "supervised": supervised,
        "unsupervised": unsupervised_pipeline(),
        "__default__": dp + supervised + unsupervised_pipeline(),
    }
