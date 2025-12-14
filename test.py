import pygame
import numpy as np
import math

class environment:
    def __init__(self, dim):
        self.height = dim[0]
        self.width = dim[1]

        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.grey = (125, 125, 125)
        pygame.display.set_caption("Differential Drive Robot - Kinematics Simulation")
        self.map = pygame.display.set_mode((self.height, self.width))

        # Font for displaying mode and info
        pygame.font.init()
        self.font = pygame.font.Font(None, 28)


class robot:
    def __init__(self, robot_skin, start_pos, width):
        # Pose
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.theta = 0.0

        # Robot geometry
        self.w = width  # Wheelbase in pixels

        # Control variables
        self.v_c = 0.0       # Linear velocity (pixels/sec)
        self.omega_c = 0.0   # Angular velocity (rad/sec)

        # Speed constants
        self.linear_accel = 50.0    # pixels/sec^2
        self.angular_accel = 1.0    # rad/sec^2
        self.max_linear_speed = 100.0  # px/s
        self.max_angular_speed = 2.0   # rad/s

        # Wrap behavior (kept for flexibility). Default False -> use bounce.
        self.wrap = False

        # Bounce energy loss factor (multiply speed on bounce)
        self.bounce_energy = 0.8

        # Load or create image
        try:
            self.img = pygame.image.load(robot_skin).convert_alpha()
        except Exception:
            # Create a simple robot image programmatically (front points to +x)
            self.img = pygame.Surface((50, 30), pygame.SRCALPHA)
            pygame.draw.rect(self.img, (0, 0, 255), (0, 0, 50, 30))
            pygame.draw.polygon(self.img, (255, 0, 0), [(40, 15), (50, 10), (50, 20)])  # Front indicator

        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

        # Mode and autonomous control
        self.mode = "keyboard"  # "keyboard", "straight", "circle"
        self.trajectory = []    # list of (x, y) tuples (floats)
        self.straight_distance = 0.0
        self.target_distance = 0.0
        self.circle_radius = 0.0
        self.circle_angle = 0.0
        self.target_angle = 0.0
        self.control_active = False

    # --- New move method (discrete key registration) ---
    def move(self, event=None):
        # Register Key control events — can be called from main event loop
        if event is not None and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                self.start_straight_motion(400)  # 400 pixels forward
            elif event.key == pygame.K_2:
                self.start_circular_motion(150)  # 150 pixels radius
            elif event.key == pygame.K_0:
                self.mode = "keyboard"
                self.control_active = False
                self.v_c = 0.0
                self.omega_c = 0.0
            elif event.key == pygame.K_w:
                # Toggle wrap mode at runtime (same as in handle_event)
                self.wrap = not self.wrap
                print(f"[robot] wrap set to {self.wrap}")

    # --- Event handling (discrete events only) ---
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                self.start_straight_motion(400)  # 400 px forward
            elif event.key == pygame.K_2:
                self.start_circular_motion(150)  # radius 150 px
            elif event.key == pygame.K_0:
                # Return to keyboard control: reset state and clear pending key events
                self.mode = "keyboard"
                self.control_active = False
                self.v_c = 0.0
                self.omega_c = 0.0
                # Reset autonomous counters
                self.straight_distance = 0.0
                self.circle_angle = 0.0
                # Clear queued KEYDOWN/KEYUP events which can cause "stuck" behavior
                pygame.event.clear(pygame.KEYDOWN)
                pygame.event.clear(pygame.KEYUP)
            elif event.key == pygame.K_w:
                # Toggle wrap mode at runtime (if True -> wrap, if False -> bounce)
                self.wrap = not self.wrap
                print(f"[robot] wrap set to {self.wrap}")

    # --- Per-frame update (continuous input + integration) ---
    def update(self, dt):
        # Continuous keyboard input only (no discrete keydown handling here)
        keys = pygame.key.get_pressed()
        if self.mode == "keyboard":
            # Linear control
            if keys[pygame.K_UP]:
                self.v_c = min(self.v_c + self.linear_accel * dt, self.max_linear_speed)
            elif keys[pygame.K_DOWN]:
                self.v_c = max(self.v_c - self.linear_accel * dt, -self.max_linear_speed)
            else:
                # natural slowdown
                if self.v_c > 0:
                    self.v_c = max(0.0, self.v_c - self.linear_accel * dt)
                elif self.v_c < 0:
                    self.v_c = min(0.0, self.v_c + self.linear_accel * dt)

            # Angular control
            if keys[pygame.K_LEFT]:
                self.omega_c = min(self.omega_c + self.angular_accel * dt, self.max_angular_speed)
            elif keys[pygame.K_RIGHT]:
                self.omega_c = max(self.omega_c - self.angular_accel * dt, -self.max_angular_speed)
            else:
                if self.omega_c > 0:
                    self.omega_c = max(0.0, self.omega_c - self.angular_accel * dt)
                elif self.omega_c < 0:
                    self.omega_c = min(0.0, self.omega_c + self.angular_accel * dt)

        # Autonomous control (if any)
        if self.mode != "keyboard" and self.control_active:
            self.autonomous_control(dt)

        # Integrate pose (Euler)
        # Compute velocity vector components
        vx = self.v_c * math.cos(self.theta)
        vy = self.v_c * math.sin(self.theta)

        # Update position using vx, vy
        self.x += vx * dt
        self.y += vy * dt
        self.theta = (self.theta + self.omega_c * dt) % (2.0 * math.pi)

        # Handle wrap or bounce / bounds crossing
        wrapped = False
        bounced = False
        screen_w, screen_h = self._get_screen_size()

        if self.wrap:
            # wrap behavior (kept for flexibility)
            if self.x < 0:
                self.x += screen_w
                wrapped = True
            elif self.x > screen_w:
                self.x -= screen_w
                wrapped = True
            if self.y < 0:
                self.y += screen_h
                wrapped = True
            elif self.y > screen_h:
                self.y -= screen_h
                wrapped = True
        else:
            # Bounce behavior: reflect velocity vector and update heading
            # If we cross a vertical wall (x < 0 or x > screen_w), invert vx
            if self.x < 0:
                self.x = 0
                vx = -vx
                bounced = True
            elif self.x > screen_w:
                self.x = screen_w
                vx = -vx
                bounced = True
            # If we cross a horizontal wall (y < 0 or y > screen_h), invert vy
            if self.y < 0:
                self.y = 0
                vy = -vy
                bounced = True
            elif self.y > screen_h:
                self.y = screen_h
                vy = -vy
                bounced = True

            if bounced:
                # apply energy loss
                vx *= self.bounce_energy
                vy *= self.bounce_energy
                # new linear speed and heading
                new_speed = math.hypot(vx, vy)
                # Avoid extremely small speeds causing problems
                if new_speed < 1e-4:
                    new_speed = 0.0
                self.v_c = new_speed
                # If speed zero, keep theta unchanged; otherwise set theta from vx,vy
                if new_speed > 0.0:
                    self.theta = math.atan2(vy, vx) % (2.0 * math.pi)
                # optionally damp angular velocity as well
                self.omega_c *= 0.5
                # reset trajectory on bounce to avoid long line
                self.trajectory = [(self.x, self.y)]

        # If not wrapped/bounced, append trajectory normally
        if not wrapped and not bounced:
            self.trajectory.append((self.x, self.y))
            if len(self.trajectory) > 1000:
                self.trajectory.pop(0)

        # Update rotated image and rect
        self.rotated = pygame.transform.rotozoom(self.img, -math.degrees(self.theta), 1.0)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

    def _get_screen_size(self):
        """Helper to read current surface size (width, height)."""
        surf = pygame.display.get_surface()
        if surf:
            w, h = surf.get_size()
        else:
            # fall back to default values if surface not available
            w, h = 1200, 800
        return w, h

    # --- Autonomous commands ---
    def start_straight_motion(self, distance):
        """Start moving forward in a straight line for `distance` pixels."""
        self.mode = "straight"
        self.control_active = True
        self.straight_distance = 0.0
        self.target_distance = float(distance)

        # Set forward velocity and zero angular velocity
        self.v_c = 80.0  # px/s
        self.omega_c = 0.0

        print(f"[robot] Starting straight motion: {distance} px")

    def start_circular_motion(self, radius):
        """Start circular motion with the given radius (px). One full loop."""
        self.mode = "circle"
        self.control_active = True
        self.circle_radius = float(radius)
        self.circle_angle = 0.0
        self.target_angle = 2.0 * math.pi

        # Base linear velocity (tangential)
        linear_vel = 60.0  # px/s

        # Calculate wheel velocities using differential drive relation
        if abs(self.circle_radius) < 1e-3:
            self.circle_radius = 1.0

        v_r = linear_vel * (1.0 + (self.w / (2.0 * self.circle_radius)))
        v_l = linear_vel * (1.0 - (self.w / (2.0 * self.circle_radius)))

        # Convert to robot linear and angular velocities
        self.v_c = 0.5 * (v_r + v_l)
        self.omega_c = (v_r - v_l) / self.w

        print(f"[robot] Starting circular motion: radius {radius} px  -> v_c={self.v_c:.2f}, omega={self.omega_c:.3f}")

    def autonomous_control(self, dt):
        """Called each frame while an autonomous command is active."""
        if self.mode == "straight":
            # Count distance traveled each frame
            self.straight_distance += abs(self.v_c * dt)
            if self.straight_distance >= self.target_distance:
                # Stop and return to keyboard mode automatically
                self.v_c = 0.0
                self.omega_c = 0.0
                self.control_active = False
                self.mode = "keyboard"
                self.straight_distance = 0.0
                print("[robot] Straight motion completed! Back to keyboard control.")

        elif self.mode == "circle":
            # Track angular displacement
            self.circle_angle += abs(self.omega_c * dt)
            if self.circle_angle >= self.target_angle:
                # Stop and return to keyboard mode automatically
                self.v_c = 0.0
                self.omega_c = 0.0
                self.control_active = False
                self.mode = "keyboard"
                self.circle_angle = 0.0
                print("[robot] Circular motion completed! Back to keyboard control.")

    # --- Utilities for UI ---
    def get_kinematics_info(self):
        v_r = self.v_c + (self.omega_c * self.w) / 2.0
        v_l = self.v_c - (self.omega_c * self.w) / 2.0

        info = {
            'MODE': self.mode.upper(),
            'X': f"{self.x:.1f}",
            'Y': f"{self.y:.1f}",
            'THETA': f"{math.degrees(self.theta):.1f}°",
            'V_LINEAR': f"{self.v_c:.1f} px/s",
            'V_ANGULAR': f"{self.omega_c:.3f} rad/s",
            'V_LEFT': f"{v_l:.1f} px/s",
            'V_RIGHT': f"{v_r:.1f} px/s",
            'WRAP': str(self.wrap)
        }

        if self.mode == "straight" and self.control_active:
            info['PROGRESS'] = f"{self.straight_distance:.1f}/{self.target_distance:.1f} px"
        elif self.mode == "circle" and self.control_active:
            info['PROGRESS'] = f"{math.degrees(self.circle_angle):.1f}°/360°"
            info['RADIUS'] = f"{int(self.circle_radius)} px"

        return info

    def draw(self, surface):
        # Draw trajectory (convert to int tuples for pygame drawing)
        if len(self.trajectory) > 1:
            pts = [(int(x), int(y)) for (x, y) in self.trajectory]
            pygame.draw.lines(surface, (255, 0, 0), False, pts, 2)

        # Draw robot sprite
        surface.blit(self.rotated, self.rect)

        # Draw wheelbase indicators
        wheel_distance = self.w / 2.0
        left_wheel_x = self.x - wheel_distance * math.sin(self.theta)
        left_wheel_y = self.y + wheel_distance * math.cos(self.theta)
        right_wheel_x = self.x + wheel_distance * math.sin(self.theta)
        right_wheel_y = self.y - wheel_distance * math.cos(self.theta)

        pygame.draw.circle(surface, (0, 255, 0), (int(left_wheel_x), int(left_wheel_y)), 5)
        pygame.draw.circle(surface, (0, 255, 0), (int(right_wheel_x), int(right_wheel_y)), 5)


# ------------------- Main program -------------------
def main():
    pygame.init()
    start = (600, 300)
    running = True
    dim = (1200, 800)
    width = 50  # wheelbase in pixels

    myenv = environment(dim)
    myrobot = robot("diffdrive.png", start, width)

    last_time = pygame.time.get_ticks()

    # Optional: use a clock to cap framerate if desired
    clock = pygame.time.Clock()
    FPS = 60

    while running:
        current_time = pygame.time.get_ticks()
        dt = (current_time - last_time) / 1000.0  # seconds
        # small safety clamp on dt (avoid huge jumps if debugger paused)
        if dt > 0.05:
            dt = 0.05
        last_time = current_time

        # Event handling (discrete events)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Call the new move() method (discrete key handling)
            myrobot.move(event)
            # Keep original handler if you want its behavior (clearing queued keys etc.)
            myrobot.handle_event(event)

        # Update robot once per frame
        myrobot.update(dt)

        # Draw everything
        myenv.map.fill(myenv.white)
        myrobot.draw(myenv.map)

        # Display kinematic info
        info = myrobot.get_kinematics_info()
        y_offset = 10
        for key, value in info.items():
            text = myenv.font.render(f"{key}: {value}", True, myenv.black)
            myenv.map.blit(text, (10, y_offset))
            y_offset += 22

        # Controls display
        controls = [
            "CONTROLS:",
            "Arrow UP/DOWN: Forward/Backward",
            "Arrow LEFT/RIGHT: Rotate",
            "1: Straight Line Motion (400px)",
            "2: Circular Motion (150px radius)",
            "0: Return to Keyboard Control",
            "W: Toggle wrap (wrap/bounce)"
        ]
        y_offset = 500
        for line in controls:
            text = myenv.font.render(line, True, myenv.black)
            myenv.map.blit(text, (10, y_offset))
            y_offset += 24

        pygame.display.flip()

        # Cap framerate (optional)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
