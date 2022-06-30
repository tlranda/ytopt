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
        'choices': ["#pragma clang loop(i1) pack array(B) allocate(malloc)", " "],
        'default_value': ' ',
        }),
    ('Categorical',
        {'name': 'p2',
        'choices': ["#pragma clang loop(i1,j1,k1,i2,j2) interchange permutation(j1,k1,i1,j2,i2)", " "],
        'default_value': ' ',
        }),
    ('Ordinal',
        {'name': 'p3',
        'sequence': ['4','8','16','20','32','50','64','80','100','128','256','512','1024','2048'],
        'default_value': '256',
        }),
    ('Ordinal',
        {'name': 'p4',
        'sequence': ['4','8','16','20','32','50','64','80','100','128','256','512','1024','2048'],
        'default_value': '256',
        }),
    ('Ordinal',
        {'name': 'p5',
        'sequence': ['4','8','16','20','32','50','64','80','100','128','256','512','1024','2048'],
        'default_value': '256',
        }),
    ('Categorical',
        {'name': 'p6',
        'choices': ["#pragma clang loop(j2) pack array(C) allocate(malloc)", " "],
        'default_value': ' ',
        }),
    ('Categorical',
        {'name': 'p7',
        'choices': ["#pragma clang loop(i1) pack array(D) allocate(malloc)", " "],
        'default_value': ' ',
        }),
    ('Categorical',
        {'name': 'p8',
        'choices': ["#pragma clang loop(j2) pack array(E) allocate(malloc)", " "],
        'default_value': ' ',
        }),
    ('Categorical',
        {'name': 'p9',
        'choices': ["#pragma clang loop(i1) pack array(F) allocate(malloc)", " "],
        'default_value': ' ',
        }),
    ]

# Based on 3mm.h NI per size
lookup_ival = {16: ('N', "MINI"), 40: ('S', "SMALL"), 110: ('SM', "SM"), 180: ('M', "MEDIUM"),
               490: ('ML', "ML"), 800: ('L', "LARGE"), 1600: ('XL', "EXTRALARGE"), 3200: ('H', "HUGE"),}
__getattr__ = polybench_problem_builder(lookup_ival, input_space, HERE, name="3mm_Problem")

