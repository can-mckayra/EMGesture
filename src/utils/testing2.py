arr = [1,2,3,4,5,6,7,8,9,10]

for start in range(0, len(arr), 5):
    end = start + 5
    print(arr[start:end])
