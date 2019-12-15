# -*- coding: utf-8 -*-

import re
from collections import defaultdict

class SteinerTreeProblem(object):
    '''
        The main purpose for this class is represent in memory the Steiner Problem's instance in memory.
    '''

    def __init__(self):
        self.nro_nodes = 0
        self.nro_edges = 0
        self.nro_terminals = 0
        self.graph = defaultdict(dict)
        self.terminals = list()

        self.name = None
        self.remark = None
        self.creator = None
        self.file_name = None


class Reader(object):
    '''
        This class parses the Steiner Tree Problem instance's file and fill the Steiner Tree Problem class above.

        Based on Bruna Osti's propose
        from: <https://github.com/brunaostii/Steiner_Tree>
    '''

    def __init__(self):
        self.STP = SteinerTreeProblem()

    def parser(self, fileName):
        self.STP.file_name = fileName

        with open(fileName, 'r') as file :
            for line in file :
                if "SECTION Comment" in line :
                    self._parser_section_comment(file)

                elif "SECTION Graph" in line :
                    self._parser_section_graph(file)

                elif "SECTION Terminals" in line :
                    self._parser_section_terminals(file)

        return self.STP

    def _parser_section_comment(self,file):
        for line in file:
            _list = re.findall(r'"(.*?)"',line)
            if "Name" in line :
                _name = _list[0] if len(_list) else "Name unviable"
                self.STP.name = _name

            elif "Creator" in line :
                _creator = _list[0] if len(_list) else "Creator unviable"
                self.STP.creator = _creator

            elif "Remark" in line : 
                remark = _list[0] if len(_list) else "Creator unviable"
                self.STP.remark = remark

            elif "END" in line:
                break

    def _parser_section_graph(self, file):
        for line in file:
            if line.startswith("E ") :
                entries = re.findall(r'(\d+)', line)
                vetor = [ e for e in entries if e.isdecimal() ]

                assert len(vetor) == 3, "The line must to have three values"
                v, w, peso = vetor

                v = int(v)
                w = int(w)
                peso = int(peso)

                self.STP.graph[v][w] = peso
                self.STP.graph[w][v] = peso

            elif line.startswith("Nodes"):
                nodes = re.findall(r'Nodes (\d+)$', line)
                self.STP.nro_nodes = int(nodes[0]) if len(nodes) else -1

            elif line.startswith("Edges"):
                edges = re.findall(r'Edges (\d+)$', line)
                self.STP.nro_edges = int(edges[0]) if len(edges) else -1

            elif "END" in line :
                break

    def _parser_section_terminals(self,file):

        for line in file:
            if line.startswith("T "):
                _string = re.findall(r"(\d+)$", line)
                v_terminal = int(_string[0]) if len(_string) == 1 else -1
                self.STP.terminals.append(v_terminal)

            elif line.startswith("Terminals"):
                terminal = re.findall(r'Terminals (\d+)$', line)
                self.STP.nro_terminals = int(terminal[0]) if len(terminal) else -1

            elif "END" in line:
                break