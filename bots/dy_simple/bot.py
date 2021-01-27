
__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'

import time

import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId

from sc2.position import Point2


class Bot(sc2.BotAI):
    """
    해병 5, 의료선 1 빌드오더를 계속 실행하는 봇
    해병은 적 사령부와 유닛중 가까운 목표를 향해 각자 이동
    적 유닛또는 사령부까지 거리가 15미만이 될 경우 스팀팩 사용
    스팀팩은 체력이 50% 이상일 때만 사용가능
    의료선은 가장 가까운 체력이 100% 미만인 해병을 치료함
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build_order = list() # 생산할 유닛 목록

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.build_order = list()
        self.evoked = dict()
        self.pos=0

    async def on_step(self, iteration: int):       
        actions = list()
        #
        # 빌드 오더 생성
        # 
        
        if len(self.build_order) == 0:
            for _ in range(5):
                self.build_order.append(UnitTypeId.MARINE)
            self.build_order.append(UnitTypeId.MEDIVAC)
        
        
        ccs = self.units(UnitTypeId.COMMANDCENTER)  # 전체 유닛에서 사령부 검색
        cc = ccs.idle
        if cc.exists:
            cc = cc.first # 실행중인 명령이 없는 사령부 검색
            cc_abilities = await self.get_available_abilities(cc)
            if AbilityId.BUILD_NUKE in cc_abilities:
                actions.append(cc(AbilityId.BUILD_NUKE))
                print(self.units(UnitTypeId.NUKE).amount,"핵생산")
            ghosts = self.units(UnitTypeId.GHOST)  # 해병 검색
            if ghosts.amount == 0:
                actions.append(cc.train(UnitTypeId.GHOST))
            else :
                
                ghost = ghosts.first
                if self.pos ==0:
                    actions.append(ghost.move(Point2((60,30))))
                    self.pos =2

                threaten = self.known_enemy_units.closer_than(15, ghost.position)
                
                # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 공격 명령 생성
                if threaten.amount > 0:
                    target = threaten.closest_to(ghost)
                    actions.append(ghost(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                    if ghost.distance_to(target.position) < 9:
                        distance = ghost.position.x - target.position.x
                        if distance > 0:
                            distance = 10 + distance
                        else:
                            distance = -10 - distance 
                        actions.append(ghost.move(Point2((ghost.position.x + distance, ghost.position.y))))

                    else : 
                        print(self.units(UnitTypeId.NUKE).amount,"핵쏘래")
                        actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=target))
                elif ghost.distance_to(Point2((60,30))) < 2:
                    print(self.units(UnitTypeId.NUKE).amount, "누크에 쏘래")
                    
                    actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.enemy_start_locations[0]))
                else:
                    print("중간으로 가")
                    actions.append(ghost.move(Point2((60,30))))


        await self.do_actions(actions)
