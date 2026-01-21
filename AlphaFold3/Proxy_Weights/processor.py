import ast

def process(filename):
    datastructure = []
    with open(filename, "r") as f:
        for line in f:
            print(line)
            lineitem={}
            fields=line.split(maxsplit=2)
            for field in fields:
                title, data = field.split("=", maxsplit=1)
                title = title.strip()
                data = data.strip()

                if title == "shape":
                    data = ast.literal_eval(data)

                lineitem[title] = data

            datastructure.append(lineitem)

    return datastructure

if __name__ == "__main__":
    p=process("parameters.txt")
    print(p)
