## Machine Learning & MLOps workflow

| Stage | What to do | Tools |
|-------|------------|-------|
| We have a ML Problem | - | - |
| Data Wrangling | Scraping data or extracting data from databases or APIs | Python & Pandas |
| Data Assessment | Assess complete data to find out what to clean and strategize | Python & Pandas |
| Data Cleaning | Clean data based on above strategy | Python & Pandas |
| EDA | Explore every column including target column and multiple columns to find relations and effect on target column | Python & Pandas |
| Another round of Data Cleaning | Clean data based on above strategy | Python & Pandas |
| Feature Engineering | Create more features if possible as understood during EDA | Python & Pandas |
| Experimentation | Perform experiments to find baseline model, best model and best hyper parameters | MLflow |
| ML Pipeline | Create reproducible ML pipeline & log model in model evaluation stage to MLflow experiment tracking | DVC |
| Register Model | Push experiment tracking model to model_registry in STAGING stage | MLflow Model Registry |
| Model_Serving | Create API around model so that it can be integrated on website and users can actually use it | FastAPI & Pydantic |
| CI | 1. Run DVC pipeline on GitHub Runner<br>2. Generate the model, log & push model to staging in model registry - so that any change in code, on git push, it should trigger DVC pipeline, and new model will be added to model_registry as staging (so that testing perform)<br>3. Test model:<br>&nbsp;&nbsp;&nbsp;- Model loading<br>&nbsp;&nbsp;&nbsp;- Model signature<br>&nbsp;&nbsp;&nbsp;- Model performance testing (like threshold > 90% or compare with existing production model)<br>4. Promote model to production<br>5. Flask app testing<br>6. Dockerize Flask app & push to ECR<br>7. Deploy docker image to AWS | github actions, docker, aws |
| CD | Once docker image is in AWS ECR, We can pull that image from ECR and deploy it to EC2 or ECS | aws ec2, codedeploy, ecs |