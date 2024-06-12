# ⚙️ Data Structure


## Data Fields

Each field of annotation is as follows: 

* `episode_id`(str): the unique identifier of this episode.
* `device_info`(dict): the detailed information of the virtual device from which the episode was collected.
    * `product`(str): the product name of the emulator.
    * `release_version`(str): the Android API level of the emulator.
    * `sdk_version`(str): the version of the software development kit used for the emulator.
    * `h`(int): the height of the device screen.
    * `w`(int): the width of the device screen.
    * `device_name`(str): the name of the virtual device, one of **Pixel Fold**, **Pixel Tablet**, **Pixel 8 Pro**, **Pixel 7 Pro**, **Medium Phone**, **Small Phone**
* `task_info`(dict): the detailed information of the task from which the episode was collected.
    * `category`(str): the category of this task, one of  **Multi_Apps**, **Web_Shopping**, **General_Tool**, **Information_Management**, **Media_Entertainment**, **Social_Sharing**
    * `app`(list[str]): the Apps used for this task. 
    * `meta_task`(str): the template for this task, e.g., "Search for the next {} and set a reminder."
    * `task`(str): the specific task created by filling in the meta-task, e.g., "Search for the next New York Fashion Week and set a reminder."
    * `instruction`(str): the detailed and rephrased version of the task, including specific tools or applications, e.g., "Utilize DuckDuckgo to find the dates for the next New York Fashion Week and then use TickTick to set a reminder for the event."
* `step_length`(int): the total number of steps in this episode.
* `steps`(list[dict]): each individual step of this episode. Including the following fields:
    * `step`(int): each step within the episode is identified by a zero-indexed step number, indicating its position in sequence within the episode. For example, if the *step* is 1, it corresponds to the second step of the episode. 
    * `screenshot`(str): the current screenshot of this step
    * `action`(str): the corresponding action of this step, one of **CLICK**, **SCROLL**, **LONG_PRESS**, **TYPE**, **COMPLETE**, **IMPOSSIBLE**, **HOME**, **BACK**
    * `info`(Union[str, list[list]]): provides specific details required to perform the action specified in the *action* field. Note that all the coordinates are normalized to the range of [0, 1000].
        * if action is *CLICK*, info contains the coordinates(x, y) to click on or one of the special keys  *KEY_HOME*, *KEY_BACK*, *KEY_RECENT*.
        * if action is *LONG_PRESS*, info contains the coordinates(x, y) for the long press.
        * if action is *SCROLL*, info contains the starting(x1, y1) and ending(x2, y2) coordinates of the scroll action.
        * if action is any other value, info is empty ("").
    * `ps`(str): provides additional details or context depending on the value of the action field.
        * if action is *COMPLETE* or *IMPOSSIBLE*: may contain any additional information from the annotator about why the task is complete or why it was impossible to complete.
        * if action is *SCROLL*: contains the complete trajectory of the scroll action.



## Data Splits
we can evaluate the in- and out-of-domain performance of Agent by splitting GUI Odyssey in two ways:

* **random_split**: randomly splitting the dataset into the training and test set with the ratio of $3:1$, 

and organizing with the training set covering a portion of apps/tasks/devices and the test set covering the remaining apps/tasks/devices:


* **task_split**: proportionally samples meta-tasks from six categories. The tasks in the test set differ significantly from those in the training set. This partitioning method allows for a robust assessment of an agent's generalization capabilities across diverse tasks. 

* **device_split**: selects episodes annotated on the *Fold Phone*, which differs significantly from other devices such as smartphones and tablets, as the test set.

* **app_split**: splits based on the apps. The apps in the test set differ significantly from those in the training set.

Each of the four classifications mentioned above has a corresponding JSON file, and the fields in each JSON file are as follows:
* `train`(list[str]): the list of annotation filenames for the training set, which are equivalent to the *episode_id*.
* `test`(list[str]): the list of annotation filenames for the test set, which are equivalent to the *episode_id*.