
import argparse
import csv
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from IPython import embed

from sc2_utils import parse_bool
from termcolor import cprint

parser = argparse.ArgumentParser('NC Fellowship 2020-eval')
# 평가 정보가 저장된 경로
parser.add_argument('root_path', type=str)
# 작업 목록
parser.add_argument('--update_bots', type=lambda x: x in ('True', 'true', '1'), default=True)
parser.add_argument('--play_games', type=lambda x: x in ('True', 'true', '1'), default=True)
parser.add_argument('--export_results', type=lambda x: x in ('True', 'true', '1'), default=True)
parser.add_argument('--publish_results', type=lambda x: x in ('True', 'true', '1'), default=True)
# 옵션
parser.add_argument('--max_rounds', type=int, default=100)
parser.add_argument('--map_name', type=str, default='NCF-2020-v4')
parser.add_argument('--timeout', type=int, default=1800)
parser.add_argument('--save_replay', type=lambda x: x in ('True', 'true', '1'), default=True)
parser.add_argument('--replay_path', type=str, default=None)
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--init_elo_rating', type=float, default=1000)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--out_repo', type=str, default='https://github.com/rex8312/NCF2020_Eval.git')
args = parser.parse_args()

# 평가시작시간
t_start = datetime.now()

# 평가용 봇과 결과가 저장되는 폴더
root_dir = Path(args.root_path)
out_dir = root_dir / 'results'
replay_dir = out_dir / 'replays'
fig_dir = out_dir / 'fig'
log_dir = out_dir / 'log'
csv_file = log_dir / 'result.csv'
system_log_file = log_dir / 'system.log'

replay_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

# 봇 저장소 목록 읽기
with (root_dir / 'config.csv').open() as f:
    reader = csv.reader(f)
    repos = [(name, repo) for name, repo in reader]
    names_, repos_ = zip(*repos)
    assert len(names_) == len(set(names_)), "중복된 bot 이름이 있음"
    assert len(repos_) == len(set(repos_)), "중복된 저장소가 이름이 있음"
    repos = {name: repo for name, repo in repos}
    
# 봇 정보 저장
Team = namedtuple('Team', 'name, class_path, repo')

teams = dict()
for name in repos.keys():
    teams[name] = Team(name, f'{root_dir}.bots.{name}.bot.Bot', repos[name])

max_rounds = args.max_rounds
assert max_rounds // 10 == max_rounds / 10
init_elo_rating = args.init_elo_rating
verbose = args.verbose
out_repo = args.out_repo
