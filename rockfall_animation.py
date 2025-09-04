import pygame
import pandas as pd
import random
import textwrap
import time
import math

data = pd.read_csv("dataset.csv")
sampled_data = pd.concat([
    data[data["alert_level"] == "Low"].sample(min(40, len(data[data["alert_level"] == "Low"])), replace=True),
    data[data["alert_level"] == "Medium"].sample(min(30, len(data[data["alert_level"] == "Medium"])), replace=True),
    data[data["alert_level"] == "High"].sample(min(20, len(data[data["alert_level"] == "High"])), replace=True)
])


pygame.init()
WIDTH, HEIGHT = 1000, 600
SIDEBAR = 350
GROUND = HEIGHT - 70
LEFT_MARGIN = 150   # Reserve space for legend

screen = pygame.display.set_mode((WIDTH + SIDEBAR, HEIGHT))
pygame.display.set_caption("Rockfall Simulation - Hackathon Demo")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 22)
big_font = pygame.font.SysFont("Arial", 44, bold=True)



def draw_legend(surface):
    legend_x = 10
    legend_y = 20
    spacing = 40

    labels = [
        ("Normal Rock", (210,180,140)),
        ("Slight Crack", (160,82,45)),
        ("Medium Crack", (139,69,19)),
        ("Severe Rockfall", (200,0,0))
    ]

    title = font.render("Rock Types", True, (0,0,0))
    surface.blit(title, (legend_x, legend_y))
    legend_y += spacing

    for text, color in labels:
        pygame.draw.rect(surface, color, (legend_x, legend_y, 30, 30))
        pygame.draw.rect(surface, (0,0,0), (legend_x, legend_y, 30, 30), 2)
        label_text = font.render(text, True, (0,0,0))
        surface.blit(label_text, (legend_x + 40, legend_y + 5))
        legend_y += spacing

def draw_person(surface, x, y, step_count=0):
    head_radius = 7
    body_height = 15
    arm_length = 10
    leg_length = 12
    jump_offset = -3 if (step_count // 5) % 2 == 0 else 3

    pygame.draw.circle(surface, (50, 50, 50), (x, y + jump_offset), head_radius)
    pygame.draw.line(surface, (50, 50, 50), (x, y + jump_offset + head_radius), (x, y + jump_offset + head_radius + body_height), 3)

    arm_swing = 5 * (-1 if (step_count // 5) % 2 == 0 else 1)
    pygame.draw.line(surface, (50,50,50), (x, y + jump_offset + head_radius + 5), (x - arm_length + arm_swing, y + jump_offset + head_radius + 5 + arm_length), 2)
    pygame.draw.line(surface, (50,50,50), (x, y + jump_offset + head_radius + 5), (x + arm_length - arm_swing, y + jump_offset + head_radius + 5 + arm_length), 2)

    leg_swing = 3 * (-1 if (step_count // 5) % 2 == 0 else 1)
    pygame.draw.line(surface, (50,50,50), (x, y + jump_offset + head_radius + body_height), (x - leg_length + leg_swing, y + jump_offset + head_radius + body_height + leg_length), 2)
    pygame.draw.line(surface, (50,50,50), (x, y + jump_offset + head_radius + body_height), (x + leg_length - leg_swing, y + jump_offset + head_radius + body_height + leg_length), 2)

def draw_house(surface, x, y):
    wall_rect = pygame.Rect(x, y + 40, 100, 60)
    pygame.draw.rect(surface, (210, 180, 140), wall_rect)
    pygame.draw.rect(surface, (0,0,0), wall_rect, 2)
    roof_points = [(x, y + 40), (x + 50, y), (x + 100, y + 40)]
    pygame.draw.polygon(surface, (150, 0, 0), roof_points)
    pygame.draw.polygon(surface, (100, 0, 0), roof_points, 2)
    door_rect = pygame.Rect(x + 40, y + 70, 20, 30)
    pygame.draw.rect(surface, (100, 50, 0), door_rect)
    pygame.draw.rect(surface, (0,0,0), door_rect, 2)
    window1 = pygame.Rect(x + 10, y + 50, 20, 20)
    window2 = pygame.Rect(x + 70, y + 50, 20, 20)
    pygame.draw.rect(surface, (173, 216, 230), window1)
    pygame.draw.rect(surface, (173, 216, 230), window2)
    pygame.draw.rect(surface, (0,0,0), window1, 2)
    pygame.draw.rect(surface, (0,0,0), window2, 2)


rocks = []
peak_x = WIDTH // 2
peak_y = 80
base_y = 180
width_factor = 400  # horizontal spread

for _ in range(2):  # more rocks
    for idx, row in sampled_data.iterrows():
        x = random.randint(peak_x - width_factor, peak_x + width_factor)
        # Ensure rock doesn't spawn inside LEFT_MARGIN
        if x < LEFT_MARGIN:
            x = LEFT_MARGIN + random.randint(0, width_factor//2)
        curve_a = (base_y - peak_y) / (width_factor**2)
        target_y = peak_y + int(curve_a * (x - peak_x)**2)
        final_size = random.randint(30,50)
        rocks.append({
            "x": x,
            "y": target_y,
            "size": 0,          # pop-up animation
            "target_size": final_size,
            "alert": row["alert_level"],
            "row": row,
            "falling": False,
            "paused": False,
            "velocity": 0,
            "warned": False,
            "shake_offset": 0,
            "shake_dir": 1,
            "alert_time": None,
            "alert_shown": False,
            "alert_display": False,
            "people_time": None,
            "crack_level": 0
        })

houses = [(200, HEIGHT-170), (450, HEIGHT-170), (700, HEIGHT-170)]
people = []
warnings = []

life_saved = False
life_saved_started = False
life_saved_scale = 0.1
step_count = 0

def spawn_people():
    new_people = []
    for hx, hy in houses:
        for j in range(3):
            new_people.append({
                "x": hx + 40 + j*10,
                "y": hy + 60,
                "running": False,
                "dir": random.choice([-1, 1]),
                "escaped": False
            })
    return new_people

def draw_cracked_rock(surface, rock):
    x, y, size = int(rock["x"] + rock["shake_offset"]), int(rock["y"]), rock["size"]
    if rock["crack_level"] == 0:
        color = (210,180,140)
    elif rock["crack_level"] == 1:
        color = (160,82,45)
    elif rock["crack_level"] == 2:
        color = (139,69,19)
    else:
        color = (200,0,0)
    pygame.draw.rect(surface, color, (x, y, size, size))
    pygame.draw.rect(surface, (0,0,0), (x, y, size, size), 2)
    for i in range(rock["crack_level"]):
        start_pos = (x + random.randint(0,size), y + random.randint(0,size))
        end_pos = (x + random.randint(0,size), y + random.randint(0,size))
        pygame.draw.line(surface, (0,0,0), start_pos, end_pos, 2)


running = True
while running:
    step_count += 1
    screen.fill((140, 178, 255))  # sky
    pygame.draw.ellipse(screen, (34, 139, 34), (0, HEIGHT - 120, WIDTH, 200))
    pygame.draw.rect(screen, (71, 60, 51), (0, HEIGHT - 70, WIDTH, 70))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    draw_legend(screen)


    for hx, hy in houses:
        draw_house(screen, hx, hy)


    for rock in rocks:
        if rock["size"] < rock["target_size"]:
            rock["size"] += 1
        if not rock["falling"] and random.random() < 0.001:
            if rock["crack_level"] < 3:
                rock["crack_level"] += 1
        if rock["crack_level"] == 3 and not rock["falling"] and not rock["warned"]:
            rock["falling"] = True
            rock["velocity"] = 4
            rock["paused"] = True
            rock["alert_time"] = time.time()
            rock["alert_shown"] = True
            rock["alert_display"] = True
            rock["people_time"] = time.time()
        if rock["paused"]:
            rock["shake_offset"] += rock["shake_dir"] * 1.5
            if abs(rock["shake_offset"]) > 6:
                rock["shake_dir"] *= -1
        if rock["falling"] or rock["paused"]:
            if rock["crack_level"] == 3:
                pygame.draw.circle(screen, (255, 255, 0),
                                   (int(rock["x"] + rock["size"]//2), int(rock["y"] + rock["size"]//2)), 60, 4)
        if rock["falling"] and not rock["paused"]:
            rock["velocity"] += 0.4
            rock["y"] += rock["velocity"]
            if rock["y"] >= GROUND:
                rock["y"] = GROUND
                rock["falling"] = False
                if not rock["warned"]:
                    rock["warned"] = True
                    row = rock["row"]
                    details = f"Rockfall at Location {row['location_id']} | Region: {row['region']} | FoS: {float(row['factor_of_safety']):.2f}"
                    warnings.append(details)
                    rock["alert_display"] = False
                    if not life_saved_started:
                        life_saved = True
                        life_saved_started = True
        draw_cracked_rock(screen, rock)


    if any(r["alert_display"] for r in rocks):
        for rock in rocks:
            if rock["people_time"] and time.time() - rock["people_time"] >= 1:
                if not people:
                    people = spawn_people()
                for p in people:
                    p["running"] = True

    all_escaped = True
    for p in people:
        if p["running"] and not p["escaped"]:
            p["x"] += p["dir"] * 5
            draw_person(screen, int(p["x"]), int(p["y"]), step_count)
            if not (0 < p["x"] < WIDTH):
                p["escaped"] = True
        elif not p["escaped"]:
            draw_person(screen, int(p["x"]), int(p["y"]), step_count)
            all_escaped = False
        if not p["escaped"]:
            all_escaped = False

    if all_escaped and people:
        for rock in rocks:
            if rock["paused"]:
                rock["paused"] = False

    if any(r["alert_display"] for r in rocks):
        alert_text = big_font.render("ROCKFALL ALERT!", True, (255,0,0))
        padding_x, padding_y = 20, 10
        rect_w = alert_text.get_width() + 2*padding_x
        rect_h = alert_text.get_height() + 2*padding_y
        rect_x = WIDTH//2 - rect_w//2
        rect_y = 10
        pygame.draw.rect(screen, (255, 150, 150), (rect_x, rect_y, rect_w, rect_h), border_radius=12)
        screen.blit(alert_text, (WIDTH//2 - alert_text.get_width()//2, rect_y + padding_y))


    pygame.draw.rect(screen, (50,50,50), (WIDTH,0,SIDEBAR,HEIGHT))
    title = font.render("Rockfall Reports", True, (255,255,255))
    screen.blit(title, (WIDTH+20, 20))
    y_offset = 60
    for msg in warnings[-8:]:
        wrapped = textwrap.wrap(msg, width=35)
        for line in wrapped:
            text_surface = font.render(line, True, (255,200,200))
            screen.blit(text_surface, (WIDTH + 20, y_offset))
            y_offset += 26
        y_offset += 10


    if life_saved:
        life_saved_scale += 0.05
        if life_saved_scale > 1.0:
            life_saved_scale = 1.0
        rect_w = int(280 * life_saved_scale)
        rect_h = int(64 * life_saved_scale)
        rect_x = WIDTH//2 - rect_w//2
        rect_y = HEIGHT//2 - rect_h//2
        pygame.draw.rect(screen, (220, 255, 210), (rect_x, rect_y, rect_w, rect_h), border_radius=22)
        msg = big_font.render("Life Saved!", True, (0,180,0))
        screen.blit(msg, (WIDTH//2 - msg.get_width()//2, HEIGHT//2 - msg.get_height()//2))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
