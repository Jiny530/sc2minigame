
__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'

import time
from sc2.data import Alert
import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId
from sc2.game_state import *
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
        self.i=0
        if self.start_location.x < 40:
            self.i=1

    async def on_step(self, iteration: int):       
        actions = list()
        #
        # 빌드 오더 생성
        # 
        
        

        ccs = self.units(UnitTypeId.COMMANDCENTER)  # 전체 유닛에서 사령부 검색
        cc = ccs.idle
        '''
        if self.alert(Alert.NuclearLaunchDetected):
            print("핵온다")
        if cc.exists:
            cc = cc.first # 실행중인 명령이 없는 사령부 검색
            cc_abilities = await self.get_available_abilities(cc)

            marines = self.units(UnitTypeId.MARINE)
            if marines.exists:

                actions.append(marines.first.move(Point2((self.start_location.x - 5, self.start_location.y))))

                threaten = self.known_enemy_units.closer_than(15,marines.first.position)

            else:
                actions.append(cc.train(UnitTypeId.MARINE))


            
            ghosts = self.units(UnitTypeId.GHOST)  # 해병 검색
            if ghosts.amount == 0:
                actions.append(cc.train(UnitTypeId.GHOST))
            else :
                    
                    
                if AbilityId.BUILD_NUKE in cc_abilities:
                    actions.append(cc(AbilityId.BUILD_NUKE))
                ghost = ghosts.first
                ghost_abilities = await self.get_available_abilities(ghost)
                if AbilityId.BEHAVIOR_CLOAKON_GHOST in ghost_abilities : 
                    actions.append(ghost(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities : 
                    actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.enemy_start_locations[0]))
                    
                if self.units(UnitTypeId.NUKE).exists:
                    print("고스트 : ",ghost.position)
                    print("에너미 : ",self.bot.enemy_start_locations[0])
                
                if self.pos ==0:
                    actions.append(ghost.move(Point2((60,30))))
                    self.pos =2

                threaten = self.known_enemy_units.closer_than(15, ghost.position)
                
                # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 공격 명령 생성
                if threaten.amount > 0:
                    target = threaten.closest_to(ghost)
                    if self.i:
                        print(target)
                    actions.append(ghost(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                    if ghost.distance_to(target.position) < 9:
                        if self.i:
                            print("얼마나 멀어져있니?",ghost.distance_to(target))
                        distance = ghost.position.x - target.position.x
                        
                    if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities : 
                        if self.i : 
                            print("핵쏘래")
                        actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=target.position))
                elif AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities : 
                    if self.i: 
                        print("누크에 쏘래")
                    actions.append(ghost(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.enemy_start_locations[0]))
                
                if self.i==0 and threaten(UnitTypeId.NUKE).amount > 0:
                    print("빨간팀 핵 : ",threaten(UnitTypeId.NUKE).amount)
                    print("빨간팀 핵 : ",threaten(UnitTypeId.NUKE).first.position)
                elif self.i==1 and self.units(UnitTypeId.NUKE).amount > 0:
                    print("우리팀 핵 : ",self.units(UnitTypeId.NUKE).amount)
                    print("우리팀 핵 : ",self.units(UnitTypeId.NUKE).first.position)
                '''
        await self.do_actions(actions)



