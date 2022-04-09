import graphviz

if __name__ == '__main__':
    with open("tree.dot") as f:
        dot_graph = f.read()
    dot=graphviz.Source(dot_graph)
    dot.view()