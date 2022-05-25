# Flask-based Model API
#### EXPLORE Data Science Academy Regression Predict

### Table of Contents

- [Flask-based Model API](#flask-based-model-api)
      - [EXPLORE Data Science Academy Regression Predict](#explore-data-science-academy-regression-predict)
    - [Table of Contents](#table-of-contents)
  - [1) Overview](#1-overview)
      - [1.1) Wait, what is an API again?](#11-wait-what-is-an-api-again)
      - [1.2) How our API will work](#12-how-our-api-will-work)
        - [Description of files](#description-of-files)
  - [2) Usage Instructions](#2-usage-instructions)
      - [2.1) Creating a copy of this repo](#21-creating-a-copy-of-this-repo)
      - [2.2) Running the API on your local machine](#22-running-the-api-on-your-local-machine)
      - [2.3) Updating the API to use your own model](#23-updating-the-api-to-use-your-own-model)
        - [Prerequisites](#prerequisites)
        - [Making the changes](#making-the-changes)
      - [2.4) Running the API on a remote AWS EC2 instance](#24-running-the-api-on-a-remote-aws-ec2-instance)
  - [3) FAQ](#3-faq)

## 1) Overview

This repository forms the basis of *Task 2* for the **Regression Predict** within EDSA's Data Science course. It hosts template code which will enable students to deploy their own developed models through a web server-based API.   

#### 1.1) Wait, what is an API again?

[![What is an API](assets/imgs/What_is_an_API.png)](https://youtu.be/s7wmiS2mSXY)


An API - or Application Programming Interface - refers to a set of procedures and protocols that allows us to send and request information between ourselves and remote applications. You can think of this as a channel of communication to a remote server using specific commands that enable you to use their applications without needing to host that functionality yourself. Many types of APIs exist, but for this predict task we are interested specifically in Web APIs. These allow us to send and receive information using web development languages, such as HTML and JSON. The video above provides a simple and intuitive explanation of how APIs operate.

#### 1.2) How our API will work

<p align='center'>
   <img src="assets/imgs/API.png"
   alt='Flask Web-server'
   width=1000px/>
   <br>
   <em>A representation of the API framework that will be created within this predict task.</em>
</p>

##### Description of files

Several files within this repository enable the correct functioning of our API. We provide a high-level description of these salient files within the table below:

| File Name              | Description                                                                          |
| :--------------------- | :--------------------                                                                |
| `api.py`               | Flask web server application definition and instantiation.                           |
| `model.py`             | Contains helper functions to separate model specific code from our API definition.   |
| `utils/request.py`     | Simple script to simulate a POST request sent to our API.                            |
| `utils/train_model.py` | Code used to train the simple model used for demonstration of the API's functioning. |

## 2) Usage Instructions

#### 2.1) Creating a copy of this repo

| ⚡️ WARNING ⚡️ |
|:--------------------|
|Do **NOT** *clone* this repository. Instead follow the instructions in this section to *fork* the repo.|

As described within the Predict instructions for the Regression Sprint, this code represents a *template* from which you can base your own model's API. As such, in order to modify the template to serve your own model (and the associated code changes which are required for this), you will need to **[fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)** this repository. Failing to do this will lead to complications when trying to work on the API remotely.


![Fork Repo](assets/imgs/fork_repo.png)  

To fork the repo, simply ensure that you are logged into your GitHub account, and then click on the 'fork' button at the top of this page as indicated within the figure above.


#### 2.2) Running the API on your local machine

As a first step to becoming familiar with our API's functioning, we recommend setting up a running instance on your own local machine.

To do this, follow the steps below by running the given commands within a Git bash (Windows), or terminal (Mac/Linux):

 1. Ensure that you have the prerequisite Python libraries installed on your local machine:

 ```bash
 pip install -U flask numpy pandas scikit-learn
 ```

 2. Clone the *forked* repo to your local machine.

 ```bash
 git clone https://github.com/{your-account-name}/load-shortfall-regression-predict-api.git
 ```  

 3. Navigate to the base of the cloned repo, and run the API web-server initialisation script.


 ```bash
 cd load-shortfall-regression-predict-api/
 python api.py
 ```

 If the web server was able to initialise successfully, the following message should be displayed within your bash/terminal session:

```
----------------------------------------
Model successfully loaded
----------------------------------------
 * Serving Flask app "api" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
```

4. Leave the web server script running within the current bash/terminal session. Open a new session, and navigate to the `utils` subfolder of the cloned repo.

```
cd {your/custom/path}/load-shortfall-regression-predict-api/utils/
```

5. Run the `request.py` script located within the utils subfolder to simulate a POST request for our running API.

```
python request.py
```


If you receive an error at this point, please ensure that the web server is still running in your original bash/terminal session. If the script ran successfully, you should receive similar output to the message shown below:



```
Sending POST request to web server API at: http://127.0.0.1:5000/api_v0.1

Querying API with the following data: 
 [8764, '2018-01-01 03:00:00', 4.6666666667, 'level_8', 0.0, 5.3333333333, 89.0, 78.0, 0.0, 3.6666666667000003, 0.0, 143.3333333333, 4.6666666667, 266.6666666667, 0.0, 0.6666666667, 0.0, 'sp25', 0.0, 0, 1020.3333333333001, 0.0, 0.0, 0.0, 0, 800.0, 800.3333333333001, 1026.6666666667, 800.0, nan, 282.48333333330004, 1030.3333333333, 284.15, 284.15, 721.0, 281.67333333330004, 53.6666666667, 284.15, 284.8166666667, 280.48333333330004, 284.19, 277.8166666667, 281.01, 283.48333333330004, 284.15, 281.15, 279.1933333333, 278.15]

Received POST response:
**************************************************
API prediction result: 9518.37493239743
The response took: 0.010888 seconds
**************************************************
```

Congratulations! You've now officially deployed your first web server API, and have successfully received a response from it.

With these steps completed, we're now ready to both modify the template code to place our own model within the API, and to host this API within an AWS EC2 instance. These processes are outlined within the sections below.  

#### 2.3) Updating the API to use your own model
| ℹ️ NOTE ℹ️ |
|:--------------------|
|We strongly encourage you to be familiar with running the API as described in Section 2.2 before attempting to use your own model.|

##### Prerequisites
Before you can update the API code-base to use your own custom model, you will need to have the following:

 - Your own `sklearn` model, trained and saved as a `.pkl` file.

    For a simple example of how to pickle your model, review the script found in `utils/train_model.py`. For further instructions, consult the *'Saving and Restoring Models in Python'* train in Athena.     

    (Note: You are not limited to use only a single model within the API. Furthermore, other `sklearn` structures which have saved parameters may be required for your model to function as well. Obviously, you are expected to handle the loading of such structures in a similar way as described within this section.)

  - Code for the data preprocessing pipeline used to train your model.

    This code should cover aspects such as data cleaning, feature engineering, feature selection, and feature transformations.

    The requirement of this code is vital as your API is built to provide a standard interface for POST requests. I.e. someone asking your API to make a prediction shouldn't have to worry about what specific features your model uses internally. Instead, anyone who sends a request with the standard features within the public dataset, should expect to receive a prediction result. This design principle makes it far easier to swap out an old model for a newer one, even if ends up using radically different features.

##### Making the changes

Once you've gathered the prerequisites from the above section, making the changes to API is relatively straight forward. It involves three steps:

1. Place your `.pkl` file within the `assets/trained-models/` directory of the repo.

2. Modify the `api.py` file by changing the `path_to_model` variable to reflect the new model `.pkl` file.

3. Modify the `model.py` file by adding your data preprocessing code to the `_preprocess_data()` helper function.  

If the following steps were carried out successfully, running the API should now produce a new prediction result.  

#### 2.4) Running the API on a remote AWS EC2 instance
| ℹ️ NOTE ℹ️ |
|:--------------------|
|You will only be able to work on this section of the API setup once you've completed the *'Introduction to Amazon AWS - Part I'* train on Athena.|

The following steps will enable you to run your web server API on a remote EC2 instance, allowing it to the queried by any device/application which has internet access.

Within these setup steps, we will be using a remote EC2 instance, which we will refer to as the ***Host***, in addition to our local machine, which we will call the ***Client***. We use these designations for convenience, and to align our terminology with that of common web server practices. In cases where commands are provided, use Git bash (Windows) or Terminal (Mac/Linux) to enter these.

1. Ensure that you have access to a running AWS EC2 instance with an assigned public IP address. Instructions for this process are found within the *'Introduction to Amazon AWS - Part I'* train on Athena.

2. Install the prerequisite python libraries on both the Host (EC2 instance), and Client (local machine):

```bash
pip install -U flask numpy pandas scikit-learn
```

3. Clone your copy of the API repo onto both the Host and Client machines, then navigate to the base of the cloned repo:

```bash
git clone https://github.com/{your-account-name}/load-shortfall-regression-predict-api-template.git
cd load-shortfall-regression-predict-api-template/
```
**[On the Host]:**

4.  Run the API web-server initialisation script.

```bash
python api.py
```

If this command ran successfully, the following output should be observed on the Host:

```
----------------------------------------
Model successfully loaded
----------------------------------------
 * Serving Flask app "api" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
```

**[On the Client]:**

5.  Navigate to the `utils` subdirectory within the repo.

```bash
cd utils/
```

6. Open the `request.py` file using your favorite text editor.

    Change the value of the `url` variable to reflect the ***public IP address*** of the Host. (Instructions for getting the public IP address are provided within the *‘Introduction to Amazon AWS - Part I’* train on Athena.)

```bash
url = 'http://{public-ip-address-of-remote-machine}:5000/api_v0.1'
```   

7. Once the editing is completed, close the file and run it:

```bash
python request.py
```

 If the command ran successfully, you should see output similar to the following:

```
Sending POST request to web server API at: http://54.229.152.221:5000/api_v0.1

Querying API with the following data: 
 [8764, '2018-01-01 03:00:00', 4.6666666667, 'level_8', 0.0, 5.3333333333, 89.0, 78.0, 0.0, 3.6666666667000003, 0.0, 143.3333333333, 4.6666666667, 266.6666666667, 0.0, 0.6666666667, 0.0, 'sp25', 0.0, 0, 1020.3333333333001, 0.0, 0.0, 0.0, 0, 800.0, 800.3333333333001, 1026.6666666667, 800.0, nan, 282.48333333330004, 1030.3333333333, 284.15, 284.15, 721.0, 281.67333333330004, 53.6666666667, 284.15, 284.8166666667, 280.48333333330004, 284.19, 277.8166666667, 281.01, 283.48333333330004, 284.15, 281.15, 279.1933333333, 278.15]

Received POST response:
**************************************************
API prediction result: 9518.37493239743
The response took: 0.010888 seconds
**************************************************
```

If you have completed the steps in 2.3), then the prediction result should differ from the one given above.

**[On the Host]**

You should also see an update to the web server output, indicating that it was contacted by the Client (the values of this string will differ for your output):

```
102.165.194.240 - - [08/June/2021 07:31:31] "POST /api_v0.1 HTTP/1.1" 200 -
```

If you are able to see these messages on both the Host and Client, then your API has successfully been deployed to the Web. Snap ⚡️!

## 3) FAQ

This section of the repo will be periodically updated to represent common questions which may arise around its use. If you detect any problems/bugs, please [create an issue](https://help.github.com/en/github/managing-your-work-on-github/creating-an-issue) and we will do our best to resolve it as quickly as possible.

We wish you all the best in your learning experience :rocket:

<p align='center'>
   <img src="assets/imgs/EDSA_logo.png"
   alt='EXPLORE Data Science Academy Logo'
   width=450px/>
   <br>
</p>
