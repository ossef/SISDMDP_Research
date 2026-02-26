from scipy.sparse import csr_matrix

class Graph:
    def __init__(self, model,city):

        #--- Read ".sz" file
        fileSz = model+city+'/'+city+'_a0-reordre.sz'
        with open(fileSz, 'r') as file:
            lines = file.readlines()
            M, N, X = int(lines[0]), int(lines[1]), int(lines[2])
            print("=> (M, N, X) = ", M, N, X)

        #--- Read ".cd" file
        fileCd = model+city+'/'+city+'_a0-reordre.cd'
        states = []
        with open(fileCd, 'r') as file:
            lines = file.readlines()
            for line in lines :
                elemts = line.split()
                id, m, d, h, x = int(elemts[0]), int(elemts[1]), int(elemts[2]), int(elemts[3]), int(elemts[4])
                states.append((m, d, h, x ))

        #--- Read ".part" file
        filePart = model+city+'/'+city+'_a0-reordre.part'
        superstates = []
        with open(filePart, 'r') as file:
            lines = file.readlines()
        nParts = int(lines[0])
        for line in lines[1:1 + nParts]:
            # format: id | first last | ...
            first_last = line.split("|")[1].strip().split()
            first = int(first_last[0])
            last = int(first_last[1])
            superstates.append(list(range(first, last + 1)))
        print(f"=> K = {nParts} superstates")

        #print("States : ",states)
        #print("Superstates : ",superstates)
        self.N      = N
        self.model  = model
        self.city   = city
        self.states = states
        self.superstates  = superstates
        self.csr_sparse   = None

    def showLine(self,id):
        print("Line[{}] = {}",id,self.csr_sparse[id])

    def showGraph(self):
        print("Graph = ",self.csr_sparse)

    def read_Rii_Matrixe(self,action):
        fileRii = self.model+self.city+'/'+self.city+'_a'+str(action)+'-reordre.Rii'
        self.csr_sparse = None
        rows, cols, data = [], [], [] 
        with open(fileRii, 'r') as file:
            lines = file.readlines()
            for line in lines: 
                elemts = line.split()
                node, degre = int(elemts[0]), int(elemts[1])
                for d in range(1,degre+1) : 
                    proba, state = float(elemts[2*d]), int(elemts[2*d+1])
                    rows.append(node)
                    cols.append(state)
                    data.append(proba)
        self.csr_sparse = csr_matrix((data, (rows, cols)), shape=(self.N, self.N))