---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 40

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

S173920, S172533, S153520

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

In our project we used the transformers framework as we worked with tweets which is a perfect problem for natural language processing. In the transformers framework we worked with the roBERTa model which was pretrained and downloaded from huggingface https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest. This model can be fine-tuned and customized to adapt to specific tasks or data, making it an ideal solution for tweets which might contain informal language or slang. With PyTorch Transformers, we can quickly and accurately classify the sentiment of tweets, making it perfect for our problem. Also, with Pytorch lightning we can quickly develop a model, and very easily train it using the lightning framework.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

We used docker for managing our dependencies. The docker file contains the relevant packages needed from apt. It also includes a file called requirements.txt which holds all the python dependencies, these are collected using pip. To get a copy of the environment the user would need to pull, build and run the docker image. 

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

We generated a cookiecutter template for our project. We use the “src” folder for our model. It contains a “data” folder from where a script downloads the data from Kaggle through their API and stores it in .csv files.
The “models” folder contains both the code for the pretrained mode, the predict model, the training model and our deployment code. The “reports” folder contains this template, and a readme.md is provided at the root of the repo, introducing the project. So are the dockerfiles and requirements, that we use in GCP.
These are the only folders being utilized in the cookiecutter structure for the project.


### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We did try to comply with the PEP 8 style guide throughout the Python code. Following some agreed upon standard usually makes it easier for others to read and debug the code for other developers. This can save time and decrease the codebase in a company. Common standard can also help catch structual mistakes in the code earlier on.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement?**
>
> Answer:

We didn’t have enough time to test all our modules thoroughly, as we ended up spending more time on the deployment and machine learning. But the structure for unittests in Pytest is set up (and demonstrated through screenshots in question 11). 

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> **Answer length: 100-200 words.**
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

In our project we have a very low code coverage (below 10%) as we didn’t implement that much unit testing. This was because of time and errors during the development process. But we understand that having a high code coverage would benefit a lot to verification of our code. Even if we had 100% we would not trust it be error free, as some error might not get caught in the test. This is defined by how good the test is written. As code coverage is a quantitative measurement, it would be better to also include test coverage as it would be a qualitive measurement for the code. 

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

While two group members worked on the model, the last one was working on the cloud setup, docker contianers etc. No branching or pull requests were used in the project because of its simplicity. Not much iterative design was made becase of the limited time.
While not ideal for larger projects, it caused no issues over the course of the five project days. However, using branches and pull requests for any project larger that this is the way to go - it eases debugging because of better version control and tracability of the code. Any new feature can be unit tested in its own branch before merging to the main project, and changes can be reverted if something goes wrong.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

DVS wasn’t used for the project, only the exercises. We didn’t have enough code iterations to require version control for the project, which is reasoning behind down prioritizing this. The group did not run into any issues with code needing to be reverted back to an earlier version. However, managing version control of large projects is essential in order to succeed. Having a well-structured version history eases the workload for code deployments and can makes debugging easier, as reverting back to functioning code is a one-line command. In this way, a developer can develop new code in the middle of a pipeline without the fear of breaking the whole system (almost) irreversibly. 

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

Our CI structure is not fully implemented for the project as the group was pressed for time. However, a simple pytest unittest is demonstrated in the dataloader "get_data.py", checking if the .csv files are collected from Kaggle through the API and extracted from the .zip file. [this figure](figures/pytestFail.png) Shows the Pytest output when a wrong path is given to the file, thus causing the module to fail. [this figure](figures/pytestPass.png) shows the same test passing when the path is correct, such that the test is passed. 
For a robust, deployed system, such unit tests would need to cover a majority of the code. It ensures a quite modular approach of the code layout which is important in large systems.
In Pytest, specific assertion errors can contain detailed descriptions, pointing the user in the right direction when issues arise. Systems with unit tests are often well suited for nightly builds, and these automated runs will oftentimes catch bugs in the codebase in unexpected places. CI can ensure that the project can be deployed accross different platforms with more ease, and other developers can get an better overview of the code with less effort. All these features can save a company a large work load and provide better tests for released software, causing fewer bugs to reach customers.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

In our project the hyperparameters was mainly hardcoded, but we would also use of simple python argparser to change the learning rate. The most efficient way to find the learning rate would be to use pytorch lightnings trainer to estimate the best learning rate. 

We didn’t use config files but it would been very beneficial to for example use Hydra for configuration of the hyperparameters 

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

To keep the projects experiments reproductive we mainly used docker. Using dockers containers we could create an environment which helped isolating the project from the underlying OS. We started using docker from the start of the development of the model, and it also helped with sharing the environment between group members.

To make the projects experiment even more reproductive we could have used config files. Config files would have helped to keep the hyperparameters precisely defined. A package such as hydra would have been perfect for keeping experiments reproductive. For a fresh group member, the config files and docker image would have been a great help, for keeping experiments simple and easy to understand.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

[Tensorboard](figures/tensorboard.png)
  
In our project we used pytorch lightning as it streamlined the development and reduced boilerplate for our project. Inside lightning we used the lightninglogs to track our training progress, where metrics as epochs and training loss was tracked. In the logs a lot more information could have been tracked like accuracy and gradients. This information can be used to monitor the performance of our model and identify any issues that may be affecting its performance. For example, if the loss is not decreasing or it could indicate if the model is underfitted or overfitted.
  
In combination with lightninglogs we also used tensorboard. Tensorboard is a convenient way to visualize the data logged by lightninglogs in a pretty graph and in real time. This means that the data logged by lightninglogs, can be analyzed and help to identify patterns. One of the biggest parts is identifying trends in our model performance such as learning rate, batch size and number of epochs.
  
Tensorboard could also been used to visualize the model architecture, which would help to understand the model, as it would able to display number of layers, neurons and activation functions used.
  
Lightninglogs and tensorboard are very powerful tools together. Another tool we could have considered using would have been weight and biases as it would have helped a lot in tuning the hyperparamaters. 

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

There were created 2 different docker images one for training and one for deployment. To run the docker image for training the user would run the command: docker run gcr.io/dtumlops-375010/gcp_vm_tester.
To run the docker image for deployment the user would have to run the command: docker run gcr.io/dtumlops-375010/gcp_test_app_final.
This was implemented so each group member could train the model in the same environment and issues with dependencies could be avoided. The image for deployment was used to deploy the model in GCP cloud run services.
A third docker image for inference was not created due to a lack of time, this was instead done locally on one of the group members laptop.


### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

When setting up the GCP service and connecting it to our FastAPI, the group ran into some issues which required more in depth debugging. Here, the logging files were used to locate our errors. For example, this approach helped u setting up the right port number in our FastAPI to connect with the GCP setup. 
The python project code was debugged mainly by inspecting log files and catching exceptions. Logging for the exercise was done on W&B, but for the project, we used Pytorch Lightning log and a bit of Pytest. The logging was especially useful when setting up the training of the model, as this step takes a while to run.
 No code profiling was done in this case, but large machine learning models can benefit from a profiling, making it possible to run optimize the code to run more efficiently.


## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used three different services for the project. Engine, Bucket, Cloud Run and Vertex AI. The engine was used to run a virtual machine which could run the code using a docker image. This was done to have a virtual machine where the model could be trained. The Bucket was used virtual storage. The model was uploaded to the cloud for the purpose of creating cloud functions. This was accomplished but besides the function from the exercise, no new functions for our model was created. Last Cloud Run was used for deployment with the use of fastAPI. The deployment of the model in GCP creates a link where a user can go and write a sentence and the user will then be given a output which describes how the sentence will be classified. An implementation of vertex AI to train the model was done as well. The model was run and training began, but after 9 hours the training failed due to lack of memory. Due to a lack of time another training was not initiated.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 50-100 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

The type of the machine used is an instance of the type n1-standard-1, with no use of GPU. The implementation of GPU would help to train the model faster, but since we intended to train using vertex AI this was not important. The machine could be used for training, but another use is that the group members could use the same machine containing the same docker image for convenient testing.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:
[Bucket](figures/Picture1.png)


### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:
[GCP container](figures/Picture2.png)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

[GCP Cloud/Picture3.png)


### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:


To deploy the model locally the following command would be called: “uvicorn src.models.deployment:app –reload”. 
To take the model from local to the cloud the model was also deployed in GCP. This was done by implementing fastAPI in the code,  creating a docker image and uploading that docker image as a service in cloud run in GCP. The service gives the link https://gcp-test-app-final-stre26cjxq-ew.a.run.app/tweet where the user can use the model. The user will be led to a website where they in the tweet column can write a sentence. The user will then get a response on how the sentence is classified (positive, negative, or neutral).


### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Monitoring was not implemented on the final project, but it is an absolute necessity for any large systems. For example, monitoring can be used to see how often the GCP deployed project is run, and if anything unexpected happens during these runs. It can be used to set up alarms to notify developers when something breaks, and if the unit testing is well implemented, it should ideally be a relatively simple task to figure out, where the issue arose. All major cloud deployment platforms, including GCP, include great monitoring options.
Another thing to monitor for is data drifting. Even the world’s best models will fail in their predictions, if the input data drifts too far away from the initial values.


### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

The overall cost of running GCP was 54 kr. The cost of running the virtual machine was 32 kr.
The remaining cost was from the use of Vertex AI for training, as well as running the service created for deploying the model in GCP.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
> *
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

First of we had a lot of disease in our group which meant that had even less time than originally. In our project the biggest struggles were getting the model to work. We had a lot of problem during the development of the model as it would work. We started using a distilBERT model, where we would train it ourselves. This proved to be a very big challenge and we changed strategy to use a pretrained model. We found a huggingface pretrained transformer model and we tried to finetune it. Here another problem emerged as we struggled a lot with finetuning the model, and getting the model saved which meant that we wasted a lot of time here. Doing the deployment of the model we also had some problems with GCP. This was mainly errors on our part, as it was first time using GCP but also that GCP is rather slow to deploy. 

The biggest improvements we could have done for this project would be to decrease time used on the model and increase time using on setting up tools and automating the process of development of the model. Our group would have benefitted a lot with using tools like github actions and hydra. 


### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:
S173920, was in charge of setting torch, transformers and mainly model stuff up.
S172533, was in charge of setting github, docker, GCP up.
S153520 was in charge of setting unit testing, GCP and github.
All members contributed to help each other and understand the process of setting all the tools up and understand how the model worked.  

