**Course Two**

# **Get Started with Python**

![][image1]

# **Instructions** 

Use this PACE strategy document to record decisions and reflections as you work through this end-of-course project. You can use this document as a guide to consider your responses and reflections at different stages of the data analytical process. Additionally, the PACE strategy documents can be used as a resource when working on future projects.

# **Course Project Recap**

Regardless of which track you have chosen to complete, your goals for this project are: 

- [ ] Complete the questions in the Course 2 PACE strategy document

- [ ] Answer the questions in the Jupyter notebook project file

- [ ] Complete coding prep work on project’s Jupyter notebook

- [ ] Summarize the column Dtypes

- [ ] Communicate important findings in the form of an executive summary

# **Relevant Interview Questions** 

Completing the end-of-course project will help you respond these types of questions that are often asked during the interview process: 

* Describe the steps you would take to clean and transform an unstructured data set. 

* What specific things might you look for as part of your cleaning process?

* What are some of the outliers, anomalies, or unusual things you might look for in the data cleaning process that might impact analyses or ability to create insights?

**Reference Guide**

This project has three tasks; the visual below identifies how the stages of PACE are incorporated across those tasks.    
![][image2]

**Data Project Questions & Considerations** 

**![][image3]PACE: Plan Stage**

* How can you best prepare to understand and organize the provided information?

  Review the project brief to fully understand the objectives and deliverables.

  Identify the structure of the dataset, including its columns, data types, and missing values.

  Research similar datasets or case studies to understand common challenges and solutions.

* What follow-along and self-review codebooks will help you perform this work?

  The course-provided codebooks and Jupyter notebooks serve as guides.

  Python libraries like `pandas` and `numpy` documentation can provide syntax references.

  Tutorials on data cleaning and transformation will be beneficial for additional guidance.

* What are some additional activities a resourceful learner would perform before starting to code?

  Explore sample datasets to practice handling missing data, identifying outliers, and summarizing data.

  Draft a checklist of tasks such as data cleaning, transformation, and summary statistic generation.

  Familiarize yourself with data visualization tools like `matplotlib` and `seaborn`.

**![][image4]PACE: Analyze Stage**	

* Will the available information be sufficient to achieve the goal based on your intuition and the analysis of the variables?

  The sufficiency of the data depends on its completeness, relevance, and accuracy. Missing values or limited observations may need to be addressed.

  Perform an initial scan to check if all key variables are present and appropriately formatted for analysis.

* How would you build summary dataframe statistics and assess the min and max range of the data? 

  Use Python’s `describe()` method in `pandas` to generate summary statistics for each column.

  Check for minimum and maximum values for numeric data to ensure they fall within expected ranges.

* Do the averages of any of the data variables look unusual? Can you describe the interval data?

  Calculate the mean for each variable and compare it to known benchmarks or expected values.

  Interval data can be described as numeric data with meaningful differences between values, but no true zero point (e.g., temperature in Celsius).

**![][image5]PACE: Construct Stage**  
**Note**: The Construct stage does not apply to this workflow. The PACE framework can be adapted to fit the specific requirements of any project.

**![][image6]PACE: Execute Stage**

* Given your current knowledge of the data, what would you initially recommend to your manager to investigate further prior to performing exploratory data analysis?

  Investigate missing values and determine if they can be imputed or if the rows/columns should be removed.

  Check for duplicate records or inconsistencies in categorical variables.

  Analyze the distribution of numeric variables for anomalies like outliers.

* What data initially presents as containing anomalies?

  Extreme values in numeric data, such as outliers that fall well beyond the typical range.

  Missing or invalid entries in categorical columns.

  Mismatched data types, such as text values in numeric columns

* What additional types of data could strengthen this dataset?

  Adding contextual data such as time stamps, geographic information, or external benchmarks for comparison.

  Including additional demographic or categorical variables to enable deeper segmentation.

  Supplementing with industry-specific metrics to enrich insights.

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAA/CAYAAABXXxDfAAACWElEQVR4Xu2Wy01DMRBFaYACqIUdzVAEOyRKYMcO0QUVICIQnzJALFiwAw2Rn5ITe+wkY3te5CMd8Zux700EvKOjwaAqZ1ffv6Vyd3aw0D7ybLcwuKW8ywUM2UJm6AJDtZRZmsEgPWW2qvByDzJjFXipJ5nVFF7mUWY2gZd4ltn3gofH7AEzmL8APDRmT5jF7AXgYTFPzl+YpynMQ9mpGB7E0sGeMBdlpyJ4CJ1LeZHdVLgcUyu/eH7fUJA9a5grJTsm4WJMrfzD4pXf+t8JXN/c7SxhrpTsmISLMXPlZeb4YllYPq8Fc6VkxyhcSpkrL4RZwnezxBTMpcmuG3AhpVY+/I63gLk02XUNDmtq5VvCXDnZeYKDmqP8KH8A5TmUc67lRXY3Lc+nOw9PeKuyu2l5j094q7K7efkwJ4SPNWCuEtndvLwQZgnfzRJTMFeJ7G5a3usTXpDdTcsLq3/sPj6/+GMzmKtEdjcv3wrmKpHdR/ltHOUPpbzAIU2tvPyRe3x6m5Svw541zJWTnSc4qKmVl//zt/c/09enl8udAP+HbyNhrpzsPMFBzVx5zteC9+Rk5wkOaubKC2GW8N0sMQVz5WTnNTicUivv9QmPXTfgQkqtfEuYS5Ndo3Ap5tzKs2MSLsYc5Q+xvMBlOqfy7FYED/FWnplislMxPChVvseLwDwp2WkreJhWvoXMockuO8FDgwxWW96vyQ57wcM9y+wm8BKPMrMpvMyTzFoFXupBZqwKL+8pszWDQVrKLF1gqBYygwsY0lLe5RYG30eePTtYSJO7g4Etf9Ja+cfizwQ/AAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAA/CAYAAABXXxDfAAACi0lEQVR4Xu3WPa7TQBSG4SyAGhpKOjokxAJYGg0FDRsDwQqgACk0UF+QuHKkWM6TGWfGcxxPpLzS2+T8zPkqZ7e7c2dVfr19/b9UZ28OA7Xo7m7x8Eh9qws88hp6wyZ41DX1lqvhIVvqbavi4z3ojavgoz3praH4WI96cwg+0rPe3oTLb0EzLMKlpZbgTLRmqcJlJS5h//LJ2Z4ozVSMi+aMwJ0RmqkIl8yZwp6UKeyJ0GyzODxnCnsuKdYjNGMWB+eMOlqst2rGLA7mnPLn44ezeq1TrLVqxiQO5RTrS9w/3YXvnGrWMxzIGXFkajZib06znmDznK1H5ubFuVbNPGJjzv2r500HylzdWqtmHrExZ8txYt0ea62aecTGnJeOy9XEeqrPWqtmPmDTnHPHyaXfU5b2LdXsYeGtp7Bfa3qXaPaq8Ptnl7/HOexLWdtfq9mrwg+WHCjWU/79/q16plazrxJ+2ufvOadYi9LsTeEjD11jp5q9OvzglP2bF2f1WqdYi9Tsi8IPRh0s1iM1e0j4I/ZcUh6+fj5oX5RmXxx+MIU9KVMcgw/+fv/ubCZCszeFH4xgGnxwwHciNPsBm5a4hP1ud5g1+BHfaNHMIza2WMIx9NSHL59sO2DfUs08YuNW5rBviWYesXFLc9hXq5lPsHlLc9hXqlnPcGBrc9hXolmTOLS1KWr/BJkxi4M9eOTfzx8nn0T7cpoxi4O9mPoTNGBfSjPO4nAvGrwkvNmKcEkvivWpZirGRT1ZEnzQTFW47JY0yyJceguaoQmX96y3h+AjPerNofhYT3rrKvhoD3rjqvj4lnrb1fCQa+otm+BR19AbusAjI/WtbvHwFt19cxhoTmfv3InlEUpQkCZ1G1VEAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAA/CAYAAABXXxDfAAAClklEQVR4Xu2bTWrcQBBGtTBkka1vMttcIefwxisfwnfIOQKBhIAXQy5jDAYvE8iEkpHoeVPVKkn9J2UevJ26+3tDVgnpuitXsvLn583JK89uDgatkXc3C4enlG81welb94FDc8r3q8FhJeWWYnBITbktK3y8BbkxC3y0Jbk1KXysRbk5CXykZbl9Fbyc1oI7kv8AvFSzFtxB2TILXkafH7veWnCPJpvc8CItfJfxvITWiH/5/Kl3gJss2Rbl7Wt3ywtCw/BS8UN4+ANwlyX7ovAwTRn/4+mXasjhcOhdGi+yUeX0vfvIgzRFPGMtfx+73vAHGOCumOxU4SHNtfEMJAx/+/KuxIdw15RsvYAHNNfEx6JDGC4S7pqSrWfwY8ul8WvCtTPc5ZHNI/zQcm18DCtcO8ddHtk8wg8tc8Vr4YL1J4a7PLJ5hB9aLom3Agas8AHtLHd5ZHMPP4ppxd/dPyxyKlxIFS+yvVq8J1zYXbw3XNhUvEYYsCR8F/FauHVG0MIF7vLK9mLxVnjszC7itXDBOhMLF7jLK9uzx2vhQxjPhNFWuMBdXtmeNT4WHsYzOhYucJdXtmeL18KJFu+Bu7yyPUv839djbyxc4A/mhbu8sr2HH1nOjR9+AIuS8Wwe4YeW3nj5mxcJn2KX8V6WnBG4yyObR/ihpSd+iXPhLo9sPoMfa241nq0X8ICmFV8a7pqSrReU+nv7FHBXTHaa8CDddXyL/1anwV2W7JuEF4TuPl7gJVuMZ5MbXtTSD8A9lmyaBS+z4kvIDVOyZRG8dJDjcsq3p2TDKnh5y3J7EvhIi3JzUvhYS3JrFvhoC3JjVvh4TbmtGBxSUm6pwn/7f2wIh6aUbzULh6+Rd28OBsXk2StX0vIPQ+29KXiZPSkAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAA/CAYAAABXXxDfAAADk0lEQVR4Xu2XvWpUQRiG9wrsRL0AG5vgv9ba2PiDXoRgKdqJprARLaIItoJm1do7sFODlWWUXIFLArYJc2AOk2dmvvnm55w9gX3haXa/+eZ9zmY1mc1WWWXQXPh0d18Lzx65UKgG7p5sWLwlvGsSYckxYIelhKXGhF1GC4ssE3YbNLx8CrDjIOGlU4Jdm4aXteDksytBOKeFnZuEl2jQyFA6BM+kYPeqcHmK2vKhHcefXPJmJOhQFC6VYGHz2qn1q95cDqGdWuiSFS6TKC2opXQ/ndThohilxXJx79F+DeikCpfEGEvcxd6n/UrRTQwPx6gVf/jjRc+/xW4P50Lk3kvHaHgwhFbclXNff7z1KijOOQnN/RY6RsOD5MTTy8mLU59q7P0SeamHhY7B8FCI1IWSmOH19mYHX8+VN6S6uNDVCw8Q7dOW5FP82vnt7ZPQ9DHQ9VA4HEJ7UerT12DP7+3+71/jPTmdDHTuw0Gi/dQttfIxeI/bja8TOvfhINFeYLDf69h3253N/SnhXQZtNzr34SDRXiCJc9ZF8wB4xmK7pX7zo3MXDoWIyYc+ucVir0NTPLSL0przsX6E7kl5abFUOKe8uyu0j7NE6uhC92L5tc1b0bI5xV1i+zhHYh0J3YvlpU8qp7hmJ+dIrCOhe7G8JVZYW1yz69iHa96sS6qjhe7V8hb7r/zG9sfm8ql92o50by7P/+Y4J2HFH3x/7klLO7Ud6T64/PnAbIibX+8nP3VJXNOR7s3k3/79vP/mz7wjVTiE5kc+tkv7AOielLd/w595d8N7z/Jo62WyOM+4SOKcDaERN9A9KW+QlkvFCc+mzs/en/PmQ8S6Ebp34RBpJe8+BHvm285P7313rhV07sNBIskbcuW18J4a6NyHgyGkB8BPf21+O/qvfw68pwY69+FgCElewj4U/qWnIfWbXQ50PhQOh6iRp5ghZ4bkfBh09cIDrdCKcY57SFN5Ex5qAaW0chLNxU14sBVn53e8h8AZLVZ8qfKnN653BS5+uee9NxS54gY6iuFhiZIypZTcRTdVuCSGWyinVC4ld9BJHS6KYcuUlNNQ83DplBUuS8GiuWWlXXw/BV2KwqUaWFwjwFnNmRh0qAqXa6FIDtylhd2bhJeUQMEWsi7s3DS8bEqw6yDhpVOAHQcNL18m7DZaWGRM2GUpYakxYIdJhCVbwrsmGxavgbuPXCgkwbOrrNI2B2lGRinNSkxoAAAAAElFTkSuQmCC>