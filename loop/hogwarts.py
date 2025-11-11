students = [
    {"name" : "Hermione", "house" : "Gryffindor", "patronus" : "otter"},
    {"name" : "Harry", "house" : "Gryffindor", "patronus" : "stag"},
    {"name" : "Ron", "house" : "Gryffindor", "patronus" : "dog"},
    {"name" : "Draco", "house" : "Slytherin", "patronus" : "none"},
]

for student in students:
    print(student['name'], student['house'], student['patronus'], sep=", ")