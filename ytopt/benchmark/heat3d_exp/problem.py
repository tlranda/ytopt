from ytopt.benchmark.base_problem import polybench_problem_builder
# Used to locate kernel for ploppers
import os
HERE = os.path.dirname(os.path.abspath(__file__))

input_space = [('Categorical',
    {'name': 'p0',
    'choices': ["#pragma clang loop(j2) pack array(A) allocate(malloc)", " "],
    'default_value': ' ',
    }),
    ('Categorical',
    {'name': 'p1',
    'choices': ["#pragma clang loop(m2) pack array(B) allocate(malloc)", " "],
    'default_value': ' ',
    }),
    ('Categorical',
    {'name': 'p2',
    'choices': ["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "],
    'default_value': ' ',
    }),
    ('Ordinal',
    {'name': 'p3',
    'sequence': ['4','8','16','20','32','50','64','80','96','100','128'],
    'default_value': '96',
    }),
    ('Ordinal',
    {'name': 'p4',
    'sequence': ['4','8','16','20','32','50','64','80','100','128','2048'],
    'default_value': '2048',
    }),
    ('Ordinal',
    {'name': 'p5',
    'sequence': ['4','8','16','20','32','50','64','80','100','128','256'],
    'default_value': '256',
    }),
    ]
# Based on 
lookup_ival = {10: ("N", "MINI", 20), 20: ("S", "SMALL", 40), 30: ("SM", "SM", 70),
               40: ("M", "MEDIUM", 100), 80: ("ML", "ML", 300), 120: ("L", "LARGE", 500),
               200: ("XL", "EXTRALARGE", 1000), 300: ("H", "HUGE", 1500),}
__getattr__ = polybench_problem_builder(lookup_ival, input_space, HERE, name="Heat3d_Problem")

