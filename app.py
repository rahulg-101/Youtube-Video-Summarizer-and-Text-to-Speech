from YoutubeSummarizer.pipeline.training_pipeline import TrainingPipeline

obj = TrainingPipeline()
data_ingestion_artifact = obj.start_data_ingestion()
print(obj.start_data_transformation(data_ingestion_artifact))
