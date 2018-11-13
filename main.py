import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def semester_avg_by_major():
    global dataset
    common_plan_dataset = dataset[dataset['Codigo curso'].astype(str).str.startswith('ING1') |
                                  dataset['Codigo curso'].astype(str).str.startswith('ING2')]
    semesters = ['ING11', 'ING1', 'ING21', 'ING22']
    data_by_major = {'INGI': [], 'INGO': [], 'INGE': [], 'INGC': []}
    for semester in range(len(semesters)):
        if semesters[semester] != 'ING1':
            semester_data = common_plan_dataset[
                common_plan_dataset['Codigo curso'].astype(str).str.startswith(semesters[semester])]
        else:
            semester_data = common_plan_dataset[
                common_plan_dataset['Codigo curso'].astype(str).str.startswith(semesters[semester]) &
                (~common_plan_dataset['Codigo curso'].astype(str).str.startswith('ING11'))]
        major_mean = (semester_data.groupby(['Especialidad del momento'])['Nota'].mean())
        for major in data_by_major:
            data_by_major[major].append(major_mean.get(major))
    legend = []
    for data in data_by_major:
        plt.plot(data_by_major[data])
        legend.append(data)
    plt.axis([0, 3, 1, 7])
    plt.autoscale(enable=True, axis='y', tight=False)
    plt.legend(legend)
    plt.xticks(np.arange(4), ('Primero', 'Segundo', 'Tercero', 'Cuarto'), rotation=30)
    plt.xlabel('Semestre')
    plt.ylabel('Promedio de Notas')
    plt.grid(True)
    plt.show()


dataset = pd.read_csv("data.csv")
# Erase every tuple that has a missing or null field
dataset = dataset.dropna()

# Erase any duplicates that might be present in the file
dataset = dataset.drop_duplicates()

# Erase every tuple in which the grade is not a number
dataset = dataset.drop(dataset[(dataset['Nota'] == 'APR') | (dataset['Nota'] == 'REP')].index)

# Erase every tuple in which the major is INGA or not yet decided (ING)
dataset = dataset.drop(dataset[(dataset['Especialidad del momento'] == 'INGA')].index)

# Change any remaining data in the 'Nota' column to a numeric value
dataset['Nota'] = dataset['Nota'].apply(pd.to_numeric)

# This function generates a graph that displays the average grade each major gets during their first two years of their
# studies.
semester_avg_by_major()


# Other graph_generation_functions were not included because of the lack of necessity of them in this code.


# Fetching all student IDs
all_students = dataset['ID'].unique()

# Fetching all courses and saving them into a dictionary
aux_courses = dataset['Codigo curso'].unique()
all_courses = {}
for course in range(len(aux_courses)):
    all_courses[aux_courses[course]] = course

# Fetching all periods and saving them into a dictionary
aux_periods = dataset['Periodo'].unique()
all_periods = {}
for period in range(len(aux_periods)):
    all_periods[aux_periods[period]] = period


# Created a list in which all student records will be saved and ready for the data frame
normalized_data = []

for student in all_students:
    student_record = [student]

    # Every tuple concerning the present student in the dataset is fetch and saved in my_dataset
    my_dataset = dataset.loc[dataset['ID'] == student]
    for course in all_courses:
        # If the student has taken the course, all information is gathered from my_dataset and saved into the
        # student_record. If the course was taken more than once, it saves every instance of it.
        if (my_dataset['Codigo curso'] == course).any():
            course_data = my_dataset.loc[my_dataset['Codigo curso'] == course]
            registered_periods = []
            for i in range(len(course_data)):
                student_record.append(all_periods[course_data.iloc[i]['Periodo']])
                student_record.append(all_courses[course])
                student_record.append(course_data.iloc[i]['Nota'])
                registered_periods.append(all_periods[course_data.iloc[i]['Periodo']])
            # Then, all data from the other semesters is saved indicating the student did not take the course.
            for period in all_periods:
                if all_periods[period] not in registered_periods:
                    student_record.append(all_periods[period])
                    student_record.append(all_courses[course])
                    student_record.append(-1)
        # If the student has not taken the course, this information in also saved in the record, indicating that,
        # in period the student got a grade of -1 in the course. This negative number indicates that the course
        # was not taken by this student.
        else:
            for period in all_periods:
                student_record.append(all_periods[period])
                student_record.append(all_courses[course])
                student_record.append(-1)

    # After all the student records are gathered, the list is added to the normalized_data.
    normalized_data.append(student_record)


# After every student has been analysed, a new data frame is created using pandas, based on the normalized_data array.
data_frame = pd.DataFrame(normalized_data)
# The normalized data is saved to the current directory with the name 'ready.csv'
data_frame.to_csv('balanced_data.csv')


# Finally, we still need to create a training set, a validation set and a test set for our dataset
# For this we initialize some variables to indicate the ratio in which to distribute the data in our dataset
probability = np.random.rand(len(data_frame))
training_mask = probability < 0.75                          # 70% of data will be training
test_mask = (probability >= 0.75) & (probability < 0.9)     # 15% of data will be testing
validatoin_mask = probability >= 0.9                        # 10% of data will be validation


# We now create our three data frames
df_training = data_frame[training_mask]
df_test = data_frame[test_mask]
df_validation = data_frame[validatoin_mask]


# And then we save them all as csv files
df_training.to_csv('training_set.csv')
df_test.to_csv('testing_set.csv')
df_validation.to_csv('validation_set.csv')
