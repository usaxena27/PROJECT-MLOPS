import kfp
from kfp import dsl


#### Components of Pipeline
def data_processing_op():
    return dsl.ContainerOp(
        name="Data Processing",
        image = "usaxena27/my-mlops-app:latest",
        command = ['python', 'src/data_processing.py']  
    )


def model_training_op():
    return dsl.ContainerOp(
        name="Model Training",
        image = "usaxena27/my-mlops-app:latest",
        command = ['python', 'src/model_training.py']  
    )



### Pipeline starts here..

@dsl.pipeline(
    name="MLOps Pipeline",
    description="End-to-end MLOps pipeline"
)
def mlops_pipeline():
    data_processing_task = data_processing_op()
    model_training_task = model_training_op()

    model_training_task.after(data_processing_task)

### RUN
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        mlops_pipeline, "mlops_pipeline.yaml"
        )