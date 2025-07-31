import ast
with open('data.txt', 'r') as f:
    data=f.read()
    print(data)
    data= ast.literal_eval(data)
    for layer_name, records in data.items():
        iters, sims = zip(*records)
        print(iters)
        print('\n\n\n\n\n',sims)