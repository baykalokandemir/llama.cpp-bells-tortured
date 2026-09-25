import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
q = """select s.value, k.gridX, k.gridY, k.gridZ, k.blockX, count(*), avg(k.end-k.start)
from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName
where s.value like '%Kernel2%' or s.value like '%splitK%' or s.value like 'mul_mat_f%' or s.value like 'mul_mat_vec_f%'
group by 1,2,3,4,5 order by count(*) desc limit 20"""
for r in db.execute(q):
    print(r[0][:60], r[1:5], r[5], round(r[6] / 1e3, 1), "us")
