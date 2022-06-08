# count the frequency of a character in a string using a dictionary
s = input("Enter String: ")
a = {}
z = s.lower()
for i in z:
    if i.lower() not in a and i !=' ':
        a[i] = z.count(i)
    else:
        pass
print("Frequency")
for ch in a:
    print(ch + ":" + str(a[ch]))
