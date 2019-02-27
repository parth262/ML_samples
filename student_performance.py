import pandas as pd

df = pd.read_csv("resources/StudentsPerformance.csv")

race_grouped = df[["race/ethnicity", "math score", "reading score", "writing score"]].groupby("race/ethnicity")
parental_grouped = df[["parental level of education", "math score", "reading score", "writing score"]].groupby(
    "parental level of education")
lunch_grouped = df[["lunch", "math score", "reading score", "writing score"]].groupby("lunch")
course_grouped = df[["test preparation course", "math score", "reading score", "writing score"]].groupby(
    "test preparation course")
gender_grouped = df[["gender", "math score", "reading score", "writing score"]].groupby("gender")
