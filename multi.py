# استيراد المكتبات اللازمة
import numpy as np  # للتعامل مع المصفوفات والحسابات الرياضية
import torch  # مكتبة PyTorch لبناء الشبكات العصبية
import torch.nn as nn  # وحدات بناء الشبكات العصبية
import torch.optim as optim  # خوارزميات التحسين
import random  # لتوليد أرقام عشوائية
from collections import deque  # قائمة ذات طرفين لتخزين الخبرات
import pygame  # مكتبة لإنشاء الألعاب والرسومات
import time  # للتحكم في التوقيت
import threading  # للتعامل مع المهام المتوازية

# تعريف الألوان المستخدمة في اللعبة باستخدام نظام RGB
BLACK = (0, 0, 0)  # اللون الأسود
WHITE = (255, 255, 255)  # اللون الأبيض
RED = (255, 0, 0)  # اللون الأحمر
GREEN = (0, 255, 0)  # اللون الأخضر
BLUE = (0, 0, 255)  # اللون الأزرق
GRID_COLOR = (40, 40, 40)  # لون الشبكة
COLORS = [(0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 0, 200)]  # ألوان متعددة للأفاعي

# تهيئة مكتبة Pygame لبدء استخدامها
pygame.init()

# تعريف الشبكة العصبية
class SnakeNN(nn.Module):
    def __init__(self, input_size=15, hidden_size=512, output_size=3):
        super(SnakeNN, self).__init__()
        # تعريف الطبقات الخطية (Linear Layers)
        self.linear1 = nn.Linear(input_size, hidden_size)  # الطبقة الأولى
        self.linear2 = nn.Linear(hidden_size, hidden_size)  # الطبقة الثانية
        self.linear3 = nn.Linear(hidden_size, hidden_size)  # الطبقة الثالثة
        self.linear4 = nn.Linear(hidden_size, output_size)  # الطبقة الأخيرة
        self.leaky_relu = nn.LeakyReLU(0.01)  # وظيفة تنشيط LeakyReLU
        
    def forward(self, x):
        # تمرير البيانات عبر الطبقات مع تطبيق وظيفة التنشيط
        x = self.leaky_relu(self.linear1(x))  # الطبقة الأولى
        x = self.leaky_relu(self.linear2(x))  # الطبقة الثانية
        x = self.leaky_relu(self.linear3(x))  # الطبقة الثالثة
        return self.linear4(x)  # الطبقة الأخيرة (الإخراج)

# تعريف الذاكرة المشتركة لتخزين الخبرات
class SharedMemory:
    def __init__(self):
        self.experiences = deque(maxlen=9000000)  # قائمة لتخزين الخبرات
        self.lock = threading.Lock()  # قفل لتجنب مشاكل التزامن
        self.training_data = {  # بيانات التدريب
            'scores': [],  # النقاط
            'epsilon': [],  # قيمة Epsilon
            'losses': [],  # الخسائر
            'avg_rewards': []  # متوسط المكافآت
        }

    def add_experience(self, experience):
        # إضافة خبرة جديدة إلى الذاكرة
        with self.lock:  # استخدام القفل لتجنب التزامن
            self.experiences.append(experience)

    def sample_batch(self, batch_size):
        # أخذ عينة عشوائية من الخبرات
        with self.lock:  # استخدام القفل لتجنب التزامن
            if len(self.experiences) < batch_size:  # إذا لم يكن هناك خبرات كافية
                return None
            return random.sample(list(self.experiences), batch_size)  # عينة عشوائية

    def add_training_data(self, score, epsilon, loss, avg_reward):
        # إضافة بيانات التدريب إلى الذاكرة
        with self.lock:  # استخدام القفل لتجنب التزامن
            self.training_data['scores'].append(score)  # إضافة النقاط
            self.training_data['epsilon'].append(epsilon)  # إضافة Epsilon
            self.training_data['losses'].append(loss)  # إضافة الخسائر
            self.training_data['avg_rewards'].append(avg_reward)  # إضافة متوسط المكافآت

# تعريف لعبة الأفعى
class SnakeGame:
    def __init__(self, width=20, height=20, cell_size=20):
        self.width = width  # عرض الشبكة
        self.height = height  # ارتفاع الشبكة
        self.cell_size = cell_size  # حجم الخلية
        
        # تهيئة الشاشة
        self.screen_width = width * cell_size  # عرض الشاشة
        self.screen_height = height * cell_size  # ارتفاع الشاشة
        self.screen = pygame.Surface((self.screen_width, self.screen_height))  # إنشاء سطح الشاشة
        
        self.reset()  # إعادة تعيين اللعبة
        
    def reset(self):
        # إعادة تعيين اللعبة إلى الحالة الأولية
        self.snake = [(self.width//2, self.height//2)]  # وضع الأفعى في المنتصف
        self.direction = (1, 0)  # الاتجاه الافتراضي (يمين)
        self.food = self._place_food()  # وضع الطعام
        self.score = 0  # إعادة تعيين النقاط
        self.steps = 0  # إعادة تعيين الخطوات
        self.game_over = False  # إعادة تعيين حالة اللعبة
        return self._get_state()  # إرجاع الحالة الحالية
    
    def _place_food(self):
        # وضع الطعام في مكان عشوائي
        while True:
            food = (random.randint(0, self.width-1), random.randint(0, self.height-1))  # موقع عشوائي
            if food not in self.snake:  # التأكد من أن الطعام ليس على الأفعى
                return food
    
    def _get_state(self):
        # الحصول على الحالة الحالية للعبة
        head = self.snake[0]  # رأس الأفعى
        danger = [False, False, False]  # المخاطر (أمام، يمين، يسار)
        
        # التحقق من المخاطر
        for i, dir_check in enumerate([
            self.direction,  # الاتجاه الحالي
            self._turn_right(self.direction),  # الدوران يمين
            self._turn_left(self.direction)  # الدوران يسار
        ]):
            next_pos = (head[0] + dir_check[0], head[1] + dir_check[1])  # الموقع التالي
            if (next_pos[0] < 0 or next_pos[0] >= self.width or  # خارج الحدود الأفقية
                next_pos[1] < 0 or next_pos[1] >= self.height or  # خارج الحدود العمودية
                next_pos in self.snake):  # الاصطدام بالأفعى
                danger[i] = True  # تم الكشف عن خطر
        
        # تحديد الاتجاه الحالي
        dir_l = self.direction == (-1, 0)  # يسار
        dir_r = self.direction == (1, 0)  # يمين
        dir_u = self.direction == (0, -1)  # أعلى
        dir_d = self.direction == (0, 1)  # أسفل
        
        # تحديد موقع الطعام بالنسبة لرأس الأفعى
        food_left = self.food[0] < head[0]  # الطعام على اليسار
        food_right = self.food[0] > head[0]  # الطعام على اليمين
        food_up = self.food[1] < head[1]  # الطعام أعلى
        food_down = self.food[1] > head[1]  # الطعام أسفل
        
        # حساب المسافة إلى الذيل في كل اتجاه
        tail_left = tail_right = tail_up = tail_down = 1.0  # القيم الافتراضية
        for segment in self.snake[1:]:  # التكرار على جسم الأفعى
            if segment[1] == head[1]:  # نفس الصف
                if segment[0] < head[0]:  # الذيل على اليسار
                    tail_left = min(tail_left, abs(head[0] - segment[0]) / self.width)
                else:  # الذيل على اليمين
                    tail_right = min(tail_right, abs(head[0] - segment[0]) / self.width)
            if segment[0] == head[0]:  # نفس العمود
                if segment[1] < head[1]:  # الذيل أعلى
                    tail_up = min(tail_up, abs(head[1] - segment[1]) / self.height)
                else:  # الذيل أسفل
                    tail_down = min(tail_down, abs(head[1] - segment[1]) / self.height)
        
        # إرجاع الحالة كـ numpy array
        return np.array([
            *danger,  # المخاطر
            dir_l, dir_r, dir_u, dir_d,  # الاتجاهات
            food_left, food_right, food_up, food_down,  # موقع الطعام
            tail_left, tail_right, tail_up, tail_down  # المسافة إلى الذيل
        ], dtype=np.float32)
    
    def _turn_right(self, direction):
        # الدوران يمين
        return (-direction[1], direction[0])
    
    def _turn_left(self, direction):
        # الدوران يسار
        return (direction[1], -direction[0])
    
    def draw(self, color=GREEN):
        # رسم اللعبة على الشاشة
        self.screen.fill(BLACK)  # تعبئة الشاشة باللون الأسود
        
        # رسم الشبكة
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.screen_width, y))
        
        # رسم الأفعى
        for i, segment in enumerate(self.snake):
            rect = pygame.Rect(
                segment[0] * self.cell_size + 1,
                segment[1] * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            pygame.draw.rect(self.screen, color if i == 0 else BLUE, rect)  # الرأس بلون مختلف
        
        # رسم الطعام
        rect = pygame.Rect(
            self.food[0] * self.cell_size + 1,
            self.food[1] * self.cell_size + 1,
            self.cell_size - 2,
            self.cell_size - 2
        )
        pygame.draw.rect(self.screen, RED, rect)
        
        # رسم النقاط
        font = pygame.font.Font(None, 24)
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (5, 5))
    
    def step(self, action):
        # تنفيذ خطوة في اللعبة بناءً على الإجراء المختار
        if action == 1:  # الدوران يمين
            self.direction = self._turn_right(self.direction)
        elif action == 2:  # الدوران يسار
            self.direction = self._turn_left(self.direction)
        
        # تحريك الرأس
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # التحقق من نهاية اللعبة
        self.game_over = (
            new_head[0] < 0 or new_head[0] >= self.width or  # خارج الحدود الأفقية
            new_head[1] < 0 or new_head[1] >= self.height or  # خارج الحدود العمودية
            new_head in self.snake  # الاصطدام بالأفعى
        )
        
        if self.game_over:
            # عقوبة الاصطدام
            penalty = -20 if new_head in self.snake else -10  # عقوبة أكبر للاصطدام بالذيل
            return self._get_state(), penalty, True  # إرجاع الحالة والعقوبة ونهاية اللعبة
        
        # تحريك الأفعى
        self.snake.insert(0, new_head)
        
        # التحقق من أكل الطعام
        reward = 0
        if new_head == self.food:
            self.score += 1  # زيادة النقاط
            reward = 10 + len(self.snake)  # مكافأة أكبر للأفعى الأطول
            self.food = self._place_food()  # وضع طعام جديد
        else:
            self.snake.pop()  # إزالة الذيل إذا لم يتم أكل الطعام
            # مكافأة صغيرة للتحرك نحو الطعام
            food_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])  # المسافة إلى الطعام
            reward = 1 / (food_dist + 1)  # مكافأة عكسية للمسافة
        
        self.steps += 1  # زيادة عدد الخطوات
        
        # عقوبة الدوران في مكان واحد
        if self.steps > 100 and self.score == 0:
            return self._get_state(), -10, True  # نهاية اللعبة
        
        return self._get_state(), reward, False  # إرجاع الحالة والمكافأة واستمرار اللعبة

# تعريف الوكيل (SnakeAgent)
class SnakeAgent:
    def __init__(self, state_size=15, action_size=3):
        self.state_size = state_size  # حجم الحالة
        self.action_size = action_size  # عدد الإجراءات
        self.gamma = 0.95  # عامل الخصم
        self.epsilon = 1.0  # قيمة Epsilon (استكشاف)
        self.epsilon_min = 0.01  # الحد الأدنى لـ Epsilon
        self.epsilon_decay = 0.995  # معدل تضاؤل Epsilon
        self.model = SnakeNN(state_size, 512, action_size)  # الشبكة العصبية
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # محسن Adam
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)  # جدولة معدل التعلم
        
    def act(self, state):
        # اختيار إجراء بناءً على الحالة
        if random.random() <= self.epsilon:  # استكشاف عشوائي
            return random.randrange(self.action_size)
        
        with torch.no_grad():  # بدون حساب التدرجات
            state = torch.FloatTensor(state).unsqueeze(0)  # تحويل الحالة إلى Tensor
            act_values = self.model(state)  # الحصول على قيم Q
            return torch.argmax(act_values).item()  # اختيار الإجراء الأفضل
        
    def replay(self, batch):
        # تدريب الشبكة العصبية باستخدام الخبرات
        states = torch.FloatTensor(np.array([i[0] for i in batch]))  # الحالات
        actions = torch.LongTensor([i[1] for i in batch])  # الإجراءات
        rewards = torch.FloatTensor([i[2] for i in batch])  # المكافآت
        next_states = torch.FloatTensor(np.array([i[3] for i in batch]))  # الحالات التالية
        dones = torch.FloatTensor([i[4] for i in batch])  # نهاية الحلقة
    
        current_q = self.model(states).gather(1, actions.unsqueeze(1))  # قيم Q الحالية
        next_q = self.model(next_states).max(1)[0].detach()  # قيم Q التالية
        target_q = rewards + (1 - dones) * self.gamma * next_q  # قيم Q الهدف
    
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)  # حساب الخسارة
        self.optimizer.zero_grad()  # إعادة تعيين التدرجات
        loss.backward()  # الانتشار العكسي
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # قص التدرجات
        self.optimizer.step()  # تحديث الأوزان
        self.scheduler.step()  # تحديث معدل التعلم
    
        if self.epsilon > self.epsilon_min:  # تضاؤل Epsilon
            self.epsilon *= self.epsilon_decay
    
        return loss.item()  # إرجاع قيمة الخسارة

# تعريف مدرب متعدد الأفاعي (MultiSnakeTrainer)
class MultiSnakeTrainer:
    def __init__(self, num_snakes=4):
        self.num_snakes = num_snakes  # عدد الأفاعي
        self.shared_memory = SharedMemory()  # الذاكرة المشتركة
        self.best_score = 0  # أفضل نتيجة
        self.total_attempts = 0  # عدد المحاولات
        
        # إعداد الشاشات
        self.game_width = 400  # عرض الشاشة
        self.game_height = 400  # ارتفاع الشاشة
        
        # إنشاء النافذة الرئيسية
        self.main_display = pygame.display.set_mode(
            (self.game_width * 2, self.game_height * 2)
        )
        pygame.display.set_caption('Multi-Snake AI Training')  # عنوان النافذة
        
        # إنشاء الألعاب والوكلاء
        self.screens = []
        for i in range(num_snakes):
            game = SnakeGame(cell_size=20)  # إنشاء لعبة جديدة
            agent = SnakeAgent()  # إنشاء وكيل جديد
            self.screens.append((game, agent))  # إضافة اللعبة والوكيل إلى القائمة

    def train(self, episodes=9000000, batch_size=32):
        try:
            running = True  # حالة التشغيل
            episode = 0  # عدد الحلقات
            font = pygame.font.Font(None, 36)  # خط لعرض النصوص
            
            while running and episode < episodes:
                # تحديث جميع الأفاعي
                for i, (game, agent) in enumerate(self.screens):
                    if game.game_over:  # إذا انتهت اللعبة
                        self.total_attempts += 1  # زيادة عدد المحاولات
                        if game.score > self.best_score:  # إذا كانت النتيجة أفضل
                            self.best_score = game.score  # تحديث أفضل نتيجة
                        state = game.reset()  # إعادة تعيين اللعبة
                    else:
                        state = game._get_state()  # الحصول على الحالة الحالية
                        
                    action = agent.act(state)  # اختيار إجراء
                    next_state, reward, done = game.step(action)  # تنفيذ الإجراء
                    
                    self.shared_memory.add_experience(  # إضافة الخبرة إلى الذاكرة
                        (state, action, reward, next_state, done)
                    )
                    
                    # تدريب الوكيل
                    batch = self.shared_memory.sample_batch(batch_size)  # أخذ عينة من الخبرات
                    if batch:
                        loss = agent.replay(batch)  # تدريب الشبكة العصبية
                        if done:
                            self.shared_memory.add_training_data(  # إضافة بيانات التدريب
                                game.score,
                                agent.epsilon,
                                loss,
                                reward
                            )
                    
                    # رسم اللعبة
                    game.draw(COLORS[i])  # رسم اللعبة بلون معين
                    x = (i % 2) * self.game_width  # حساب الموضع الأفقي
                    y = (i // 2) * self.game_height  # حساب الموضع العمودي
                    self.main_display.blit(game.screen, (x, y))  # عرض اللعبة على الشاشة
                
                # رسم جميع الشاشات
                for i, (game, _) in enumerate(self.screens):
                    x = (i % 2) * self.game_width
                    y = (i // 2) * self.game_height
                    self.main_display.blit(game.screen, (x, y))
                
                # عرض الإحصائيات
                stats_text = f"Best Score: {self.best_score} | Attempts: {self.total_attempts}"
                text_surface = font.render(stats_text, True, WHITE)
                self.main_display.blit(text_surface, (20, 20))
                
                pygame.display.flip()  # تحديث الشاشة
                
                # التعامل مع الأحداث
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:  # إذا تم إغلاق النافذة
                        running = False
                
                episode += 1  # زيادة عدد الحلقات
                time.sleep(0.01)  # تأخير صغير لجعل اللعبة مرئية
                
        finally:
            pygame.quit()  # إغلاق Pygame

# بدء التدريب
if __name__ == "__main__":
    trainer = MultiSnakeTrainer()  # إنشاء المدرب
    trainer.train()  # بدء التدريب