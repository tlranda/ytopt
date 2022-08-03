from ytopt.benchmark.base_problem import ecp_problem_builder
from ytopt.benchmark.base_plopper import ECP_Plopper
# Used to locate kernel for ploppers
import os
HERE = os.path.dirname(os.path.abspath(__file__))

input_space = [('Ordinal',
    {'name': 'p0',
     'sequence': ['2','4','8','16','32','48','64','96','128','192','256'],
     'default_value': '128',
    }),
    ('Ordinal',
    {'name': 'p1',
     'sequence': ['100','200','400','640','800','1000','1280','1600','2000'],
     'default_value': '1000',
    }),
    ('Categorical',
    {'name': 'p2',
     'choices': ["#pragma clang loop unrolling full", " "],
     'default_value': ' ',
    }),
    ('Categorical',
    {'name': 'p3',
     'choices': ["#pragma omp parallel for", " "],
     'default_value': ' ',
    }),
    ('Ordinal',
    {'name': 'p4',
     'sequence': ['2','4','8','16','32','64','96','128','256'],
     'default_value': '96',
    }),
    ('Ordinal',
    {'name': 'p5',
     'sequence': ['2','4','8','16','32','64','96','128','256'],
     'default_value': '256',
    }),
    ('Ordinal',
    {'name': 'p6',
     'sequence': ['10','20','40','64','80','100','128','160','200'],
     'default_value': '100',
    }),
    ('Categorical',
    {'name': 'p7',
     'choices': ['compact','scatter','balanced','none','disabled', 'explicit'],
     'default_value': 'none',
    }),
    ('Categorical',
    {'name': 'p8',
     'choices': ['cores','threads','sockets'],
     'default_value': 'cores'
    }),
    ]
class RSBench_Plopper(ECP_Plopper):
    def runString(self, outfile, dictVal, *args, **kwargs):
        d_size = args[0]
        return f"srun -n 1 {outfile[:-len(self.output_extension)]} -s large -m event -l {d_size}"
# Based on
lookup_ival = {100000: ("S", "SMALL"), 500000: ("SM", "SM"), 1000000: ("M", "MEDIUM"),
               2500000: ("ML", "ML"), 5000000: ("L", "LARGE"), 10000000: ("XL", "EXTRALARGE")}
__getattr__ = ecp_problem_builder(lookup_ival, input_space, HERE, name="RSBench_Problem", plopper_class=RSBench_Plopper)


