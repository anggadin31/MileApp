import sys
import time
from jobs import MileAppML

jobs = {
    'MILE_APP': MileAppML,
}

if __name__ == '__main__':
    _job_start = time.time()
    job_name = sys.argv[1].upper()
    jobs[job_name]().run()
    print("Total time of '{}' execution: {} minute(s)".format(job_name, round((time.time() - _job_start) / 60, 2)))