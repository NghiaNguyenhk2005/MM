from policy import Policy
from policy import GreedyPolicy
import random
import numpy as np
import math 

#Giải thuật luyện thép
class SimulatedAnnealingPolicy(Policy):
    def __init__(self, initial_temperature=1000, cooling_rate=0.95, iterations=1000):
        # Khởi tạo các tham số của giải thuật Simulated Annealing
        self.initial_temperature = initial_temperature  # Nhiệt độ ban đầu
        self.cooling_rate = cooling_rate                # Hệ số làm nguội
        self.iterations = iterations                    # Số lần lặp

    def get_action(self, observation, info):
        # Khởi tạo các biến cho giải thuật
        best_action = None
        best_cost = float('inf')
        temperature = self.initial_temperature

        # Sử dụng giải thuật tham ăn để tạo giải pháp ban đầu
        greedy_action = GreedyPolicy().get_action(observation, info) 
        if greedy_action is None:
            return None                 # Trường hợp không tìm thấy giải pháp
        
        # Gán giải pháp ban đầu cho giải thuật
        current_action = greedy_action
        current_cost = self.calculate_cost(current_action, observation) 

        # Giải thuật Simulated Annealing để tìm ra giải pháp tốt nhất 
        # dựa trên các giải pháp láng giềng
        for _ in range(self.iterations):
            # Tạo giải pháp láng giềng
            neighbor_action = self.generate_neighbor(current_action, observation)
            if neighbor_action is None:
              continue          # Bỏ qua lần lặp này nếu không có

            # Tính toán sự khác biệt về chi phí
            neighbor_cost = self.calculate_cost(neighbor_action, observation)
            delta_cost = neighbor_cost - current_cost

            # Điều kiện chấp nhận láng giềng
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                current_action = neighbor_action
                current_cost = neighbor_cost

            # Cập nhật giải pháp tốt nhất
            if current_cost < best_cost:
                best_cost = current_cost
                best_action = current_action

            # Giảm nhiệt độ
            temperature *= self.cooling_rate

        return best_action


    def generate_neighbor(self, action, observation):
        # Tạo giải pháp láng giềng bằng cách thay đổi vị trí của một sản phẩm ngẫu nhiên
        new_action = action.copy()
        new_x = max(0, action["position"][0] + random.randint(-1,1))
        new_y = max(0, action["position"][1] + random.randint(-1,1))

        new_action["position"] = (new_x, new_y)

        # Kiểm tra vị trí có hợp lý?
        stock = observation["stocks"][new_action["stock_idx"]]
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = new_action["size"]
        if new_x + prod_w <= stock_w and new_y + prod_h <= stock_h and self._can_place_(stock, new_action["position"], new_action["size"]):
            return new_action
        else:
            return None # Trả về None nếu không có giải pháp láng giềng hợp lý


    def calculate_cost(self, action, observation):
        # Tính chi phí dựa trên diện tích còn dư
        stock = observation["stocks"][action["stock_idx"]] # Lấy stock
        prod_area = action["size"][0] * action["size"][1]  # Diện tích sản phẩm
        stock_area = np.sum(stock != -1)  # Tính diện tích đã sử dụng trên stock
        return stock_area - prod_area  # Chi phí là diện tích còn dư
        
