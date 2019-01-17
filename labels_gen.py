class labels_gen:
    def generate(self,pathAY,pathAN):
        lines_Yauto = sum(1 for l in open(pathAY))
        lines_Nauto = sum(1 for l in open(pathAN))

        labels = [1] * (lines_Yauto) + [0] * (lines_Nauto)
        
        return labels