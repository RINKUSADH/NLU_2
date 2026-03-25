import random

first_names = ["Aadhya", "Aarav", "Aarnav", "Aarohi", "Aaryan", "Aayush", "Aditi", "Aditya", "Advik", "Ahana", "Akash", "Akhil", "Akshay", "Amira", "Amit", "Amrita", "Ananya", "Aniket", "Anil", "Anjali", "Ansh", "Anushka", "Arav", "Ariana", "Arjun", "Armaan", "Arnav", "Aryan", "Atharv", "Avani", "Avni", "Ayaan", "Ayush", "Bhavya", "Chaitanya", "Chirag", "Daksh", "Deepak", "Dev", "Dhruv", "Dia", "Diya", "Esha", "Gauri", "Gautam", "Gia", "Hari", "Harsh", "Hrithik", "Ira", "Ishan", "Ishana", "Ishant", "Ishita", "Kabir", "Karan", "Karthik", "Kavya", "Khushi", "Kian", "Krish", "Krishna", "Kritika", "Kunal", "Laksh", "Lakshay", "Madhav", "Mahika", "Manish", "Meera", "Meghna", "Mihir", "Mira", "Misha", "Myra", "Navya", "Neel", "Neha", "Nikhil", "Nisha", "Nishant", "Nitya", "Ojas", "Ojasvi", "Om", "Pari", "Parth", "Pooja", "Pranav", "Prashant", "Prateek", "Priya", "Rahul", "Raj", "Rajat", "Rajesh", "Rakesh", "Ramesh", "Ravi", "Ria", "Rishabh", "Rishi", "Riya", "Rohan", "Rohit", "Rudra", "Rutuja", "Sachi", "Sahil", "Samar", "Sameer", "Samir", "Sana", "Sanika", "Sanjay", "Sanjeev", "Sanya", "Sara", "Sarthak", "Shaurya", "Shiv", "Shivam", "Shivansh", "Shlok", "Shreya", "Shruti", "Sia", "Siddharth", "Simran", "Sneha", "Sohan", "Suhana", "Sunil", "Suresh", "Surya", "Tahira", "Tanisha", "Tanya", "Tara", "Uday", "Uma", "Utkarsh", "Vaishnavi", "Vansh", "Varun", "Ved", "Vedant", "Veer", "Vidhi", "Vidyut", "Vihaan", "Vijay", "Vikas", "Vikram", "Vivaan", "Yash", "Yuvraj", "Zara", "Zoya"]

last_names = ["Agarwal", "Ahuja", "Arora", "Bansal", "Bhatia", "Bhatt", "Chakraborty", "Chatterjee", "Chauhan", "Chopra", "Das", "Desai", "Deshmukh", "Dixit", "Dubey", "Garg", "Ghosh", "Goel", "Goyal", "Gupta", "Iyer", "Jain", "Jha", "Joshi", "Kapoor", "Kaur", "Khan", "Khanna", "Khatri", "Khurana", "Kishore", "Kumar", "Mahajan", "Malik", "Malhotra", "Mathur", "Mehra", "Mehta", "Menon", "Mishra", "Mittra", "Nair", "Nanda", "Narang", "Nath", "Nayak", "Pandey", "Patel", "Pathak", "Patil", "Pillai", "Prasad", "Pujari", "Rajput", "Raman", "Rao", "Rastogi", "Rathore", "Reddy", "Roy", "Sahni", "Sahu", "Saini", "Sarkar", "Sen", "Sethi", "Shah", "Sharma", "Shetty", "Shukla", "Singh", "Sinha", "Soni", "Sood", "Srivastava", "Sur", "Swami", "Tandon", "Thakur", "Tiwari", "Upadhyay", "Varma", "Verma", "Vyas", "Yadav"]

names = set()
while len(names) < 1000:
    first = random.choice(first_names)
    last = random.choice(last_names)
    names.add(f"{first} {last}")

with open("TrainingNames.txt", "w") as f:
    for name in list(names)[:1000]:
        f.write(name + "\n")
print("Generated 1000 Indian names in TrainingNames.txt")
