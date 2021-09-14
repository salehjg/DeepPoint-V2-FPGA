# Analyzing Profiled Runs
Running the main executable of the classifier or the OpenCl unit tests produces a JSON file containing the profiled 
details regarding the host and the device layer and kernel executions.

Although the JSON file (`profiler.json`) is human-readable, it has been decided to provide a Python script to make 
the profiling procedure more user friendly. 

The Python script is located at `scripts/Report.py`. Use it as below:
```
$ python3 <RepoDir>/scripts/Report.py profiler.json
```

This command will create a new directory using the current time and date and outputs the reports in text formatted files.