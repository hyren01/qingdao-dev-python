from jdqd.a04.relation.relation_pt.algor import r_parallel
from jdqd.a04.relation.relation_pt.algor import r_choice
from jdqd.a04.relation.relation_pt.algor import r_further
from jdqd.a04.relation.relation_pt.algor import r_assumption
from jdqd.a04.relation.relation_pt.algor import r_then
from jdqd.a04.relation.relation_pt.algor import r_hypernym
from jdqd.a04.relation.relation_pt.algor import r_condition
from jdqd.a04.relation.relation_pt.algor import r_contrast
from jdqd.a04.relation.relation_pt.algor import r_causality
from tqdm import tqdm
import jdqd.a04.relation.relation_pt.algor.relation_combine as relation_combine
import jdqd.a04.relation.relation_pt.algor.relation_util

articles_dir = 'C:/work1/qingdao/archive/articles/shizheng'
import os

articles_fp = os.listdir(articles_dir)
articles_fp = [os.path.join(articles_dir, a) for a in articles_fp]

relations = [r_causality, r_condition, r_assumption, r_contrast, r_parallel,
             r_choice, r_further, r_then]

n_articles = 0
stat = {}
accu = {r.__name__: 0 for r in relations}
for a in tqdm(articles_fp[:10000]):
# for a in articles_fp[:5000]:
    with open(a, 'r', encoding='utf-8') as f:
        content = f.read()

    sentences = relation_util.split_article_to_sentences(content)

    for s in sentences:
        s = s.replace(' ', '').replace('\t', '').replace('\n', '')
        for r in relations:
            rst = relation_combine.extract_all_rules(s, r.rules,
                                                     r.keyword_rules)
            if rst:
                accu[r.__name__] = accu[r.__name__] + 1

    n_articles += 1
    if n_articles % 2000 == 0:
        stat[n_articles] = accu
print(stat)

# if __name__ == '__main__':
#
#     s = '由于陈水扁之前表示，海外账户6亿元款项是选举剩余款，与上午说法差异颇大，因此，检方还要进一步查证扁珍的说法。'
#     rst = relation_combine.extract_all_rules(s, r_causality.rules, r_causality.keyword_rules)
#     print(rst)
