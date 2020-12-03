#!/usr/bin/env python

# Copyright 2018-2020 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import nbconvert
import numpy as np
import yaml

with open("project3.ipynb") as f:
    exporter = nbconvert.PythonExporter()
    python_file, _ = exporter.from_file(f)


with open("project3.py", "w") as f:
    f.write(python_file)

from project3 import Project3


def get_gold_pressures():

    return np.load('pressure_gold.npy')

def get_gold_saturations():

    return np.load('saturation_gold.npy')

class TestSolution(unittest.TestCase):

    def setUp(self):

        with open('inputs.yml') as f:
            self.inputs = yaml.load(f, yaml.FullLoader)

    def test_project3_test_1(self):        
        
        test = Project3(self.inputs)
        test.solve()
        
        np.testing.assert_allclose(test.p, 
                                   get_gold_pressures(),
                                   atol=30.0)
        
        return

    def test_project3_test_2(self):         
        
        test = Project3(self.inputs)
        test.solve()
        
        np.testing.assert_allclose(test.saturation, 
                                   get_gold_saturations(),
                                   atol=0.02)
        
        return


if __name__ == '__main__':
    unittest.main()
