import networkx as nx
import numpy as np
import scipy
import time
import json

class GraphInterface():
    
    #去除非ASCII(防止转Graphml，XML报错)
    def stripNonChar(self,string):
        stripped = (c for c in string if 'a'<=c<='z' or 'A'<=c<='Z' or '0'<=c<='9' or c==' ')
        return ''.join(stripped)
    
    def graph2graphml(self,G,out_path):
        print('Transforming graph...')
        tik = time.time()
        nx.write_graphml(G,out_path)
        tik = time.time() - tik
        print('Transform graph to graphml successfully!(%.3fsec)'%tik)
        print()
    
    #读取DBLP
    def DBLPjson2networkxGraph(self,path,haveName=False,name='None',size=-1):
        tik = time.time()
        if haveName==True:
            G = nx.Graph(name=name)
        else:
            G = nx.Graph()
        with open(path,'r',encoding='utf-8') as f:
            counter = 0
            for line in f:
                if size != -1:
                    if counter < size:
                        counter +=1
                    else:
                        break
                jsonObj = json.loads(line)
                paperId = jsonObj.get('id')
                title = jsonObj.get('title')
                G.add_node(paperId,nodeType='paper',id=paperId,title=title,name=title)
                authors = jsonObj.get('authors')
                #有多个作者
                for person in authors:
                    G.add_node(person,nodeType='author',paperTitle=title)
                    G.add_edge(paperId,person,edgeType='paper_author',weight=1)
                year = jsonObj.get('year')
                G.add_node(paperId,year=year)
                conference = jsonObj.get('venue')
                if conference != '':
                    G.add_node(conference,nodeType='conference')
                    G.add_edge(paperId,conference,edgeType='paper_conference',weight=1)
                references = jsonObj.get('references')
                if references is not None:
                    for reference in references:
                        if G.has_node(reference):
                            G.add_edge(paperId,reference,edgeType='paper_paper',weight=1)
        f.close()
        tik = time.time()-tik
        print('Graph imported!(%.3fsec)'%tik)
        print()
        return G

    #读取ACMtxt
    def ACMtxt2networkxGraph(self,path,haveName=False,name='None'):
        tik = time.time()
        counter = 0
        if haveName==True:
            G = nx.Graph(name=name)
        else:
            G = nx.Graph()
        with open(path,'r') as f:
            for line in f:
                if line == '' or line == '\n':
                    counter = 0
                    continue
                if counter == 0:
                    title = self.stripNonChar(line[2:].rstrip())
                    G.add_node(title,nodeType='paper')
                    counter += 1
                    continue
                elif counter == 1:
                    author = self.stripNonChar(line[2:].rstrip())
                    author = author.split(',')
                    #有多个作者
                    for person in author:
                        person = person.lstrip()
                        G.add_node(person,nodeType='author')
                        G.add_edge(title,person,edgeType='paper_author')
                    counter += 1
                    continue
                elif counter == 2:
                    year = self.stripNonChar(line[2:].rstrip())
                    G.add_node(title,year=year)
                    counter += 1
                    continue
                elif counter == 3:
                    conference = self.stripNonChar('conference: ' + line[2:].rstrip())
                    G.add_node(conference,nodeType='conference')
                    G.add_edge(title,conference,edgeType='paper_conference')
                    counter += 1
                    continue
                elif counter == 4:
                    index = self.stripNonChar(line[1:].rstrip())
                    G.add_node(title,index=index)
                    counter += 1
                    continue
        f.close()
        tik = time.time()-tik
        print('Graph imported!(%.3fsec)'%tik)
        print()
        return G
    
    def exportDBLPFeature(self,in_path,out_path,size=-1,nonAbstract=True):
        tik = time.time()
        counter = 0
        f2 = open(out_path,'w')
        with open(in_path,'r') as f:
            counter = 0
            for line in f:
                if size != -1:
                    if counter < size:
                        counter +=1
                    else:
                        break
                jsonObj = json.loads(line)
                title = jsonObj.get('title')
                if title is not None:
                    title = self.stripNonChar(title.lower())
                    print(title,file=f2,end=' ')
                if nonAbstract == False:
                    abstract = jsonObj.get('abstract')
                    if abstract is not None:
                        abstract = self.stripNonChar(abstract.lower())
                        print(abstract,file=f2,end=' ')
        f.close()
        f2.close()
        tik = time.time()-tik
        print('Export DBLPFeature successfully!(%.3fsec)'%tik)
        print()
    
    #慢的话可以把大小写，ASCII去了    
    def getSubgraph(self,G,calSubK=[1,1,1],weight=[1,2,3]):
        H = nx.Graph()
        print('Calculating subgraph...')
        tik = time.time()
        for u,nbrdict in G.adjacency():
            if G.node[u]['nodeType'] is not 'author':
                continue
            H.add_node(u,nodeType='author',paperTitle=self.stripNonChar(G.node[u]['paperTitle'].lower()))
            for nVertice in nbrdict:
                for nnVertice in G.neighbors(nVertice):
                    if nnVertice == u:
                        continue
                    if calSubK[2] == 1:
                        if G.node[nnVertice]['nodeType'] == 'conference':
                            for nnnVertice in G.neighbors(nnVertice):
                                if nnnVertice == nVertice:
                                    continue
                                for nnnnVertice in G.neighbors(nnnVertice):
                                    if nnnnVertice == nnVertice:
                                        continue
                                    if G.node[nnnnVertice]['nodeType'] == 'author':
                                        #大小写ASCII
                                        H.add_node(nnnnVertice,nodeType='author',paperTitle=self.stripNonChar(G.node[nnnnVertice]['paperTitle'].lower()))
                                        H.add_edge(u,nnnnVertice,edgeType='paper_paper',weight=weight[2])
                    if calSubK[1] == 1:
                        if G.node[nnVertice]['nodeType'] == 'paper':
                            for nnnVertice in G.neighbors(nnVertice):
                                if nnnVertice == nVertice:
                                    continue
                                if G.node[nnnVertice]['nodeType'] == 'author':
                                    #大小写ASCII
                                    H.add_node(nnnVertice,nodeType='author',paperTitle=self.stripNonChar(G.node[nnnVertice]['paperTitle'].lower()))
                                    H.add_edge(u,nnnVertice,edgeType='paper_paper',weight=weight[1])
                    if calSubK[0] == 1:
                        if G.node[nnVertice]['nodeType'] == 'author':
                            #大小写ASCII
                            H.add_node(nnVertice,nodeType='author',paperTitle=self.stripNonChar(G.node[nnVertice]['paperTitle'].lower()))
                            H.add_edge(u,nnVertice,edgeType='paper_paper',weight=weight[0])
        tik = time.time() - tik
        print('Get subgraph successfully!(%.2fsec)'%tik)
        return H
    
    def exportACMFeature(self,in_path,out_path):
        tik = time.time()
        counter = 0
        f2 = open(out_path,'w')
        with open(in_path,'r') as f:
            for line in f:
                if line == '' or line == '\n':
                    counter = 0
                    continue
                if counter == 0:
                    title = self.stripNonChar(line[2:].rstrip())
                    print(title,file=f2,end=' ')
                    counter += 1
                    continue
        f.close()
        f2.close()
        tik = time.time()-tik
        print('Export ACMFeature successfully!(%.3fsec)'%tik)
        print()

    def getPInKOrder(self,G,K=300):
        tik = time.time()
        print('Calculating P K-order...')
        H = np.zeros([K,G.order(),G.order()])
        P = nx.adjacency_matrix(G)
        P = P.toarray()
        P = P.astype(float)
        for i in range(P.shape[0]):
            sumP = sum(P[i,:])
            if sumP == 0:
                continue
            sumPVec = np.full(G.order(),sumP)
            P[i,:] = np.divide(P[i,:],sumPVec)
        print('1-order')
        for i in range(K):
            H[i,:,:] = P
        for i in range(1,K):
            print('%d-order'%(i+1))
            H[i,:,:] = H[i-1,:,:].dot(P)
        tik = time.time() - tik
        print('Got P Maxtrices in K-order(%.2fsec)'%tik)
        return H
        
