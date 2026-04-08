import tkinter as tk
from tkinter import ttk
import ray
import os


class LeaderboardGUI:
    def __init__(self, actor_handle):
        self.actor = actor_handle
        self.root = tk.Tk()
        self.root.title("Casino Live Leaderboard")
        # self.root.geometry("680x450")

        # Get the current screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set the window to fill the screen, starting at the top-left corner (0,0)
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")

        # Attempt to natively "maximize" the window (handles taskbars better)
        try:
            self.root.state('zoomed')  # Works on Windows and most Linux
        except tk.TclError:
            pass  # Mac OS fallback is handled perfectly by the geometry line above

        # Get default button color to ensure cross-OS compatibility
        dummy_btn = tk.Button(self.root)
        self.default_btn_bg = dummy_btn.cget("bg")
        self.default_btn_hl = dummy_btn.cget("highlightbackground")  # ---> NEW: Save default highlight
        dummy_btn.destroy()

        # State trackers
        self.actual_tables = 0
        self.actual_trainers = 0
        self.current_target_tables = 0
        self.current_target_trainers = 0

        # Main Label
        self.label = tk.Label(self.root, text="Live Standings", font=("Arial", 16, "bold"))
        self.label.pack(pady=10)

        # Treeview for standings
        self.tree = ttk.Treeview(self.root, columns=("Rank", "ID", "Games", "Total", "Recent Avg", "Status"),
                                 show='headings')
        self.tree.heading("Rank", text="Rank")
        self.tree.heading("ID", text="Player ID")
        self.tree.heading("Games", text="Games")
        self.tree.heading("Total", text="All-Time")
        self.tree.heading("Recent Avg", text="Last 10 Avg")
        self.tree.heading("Status", text="Status")

        self.tree.column("Rank", width=50)
        self.tree.column("Games", width=60)
        self.tree.column("Status", width=220)
        self.tree.pack(expand=True, fill='both', padx=10, pady=(10, 0))

        # --- Control Panel Frame ---
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(fill='x', padx=10, pady=10)

        # 1. Tables Control Sub-frame (Moved to the far left)
        self.frame_tables = tk.Frame(self.control_frame)
        self.frame_tables.pack(side='left', padx=10)
        tk.Label(self.frame_tables, text="Tables:").pack(side='left')

        self.btn_sub_table = tk.Button(self.frame_tables, text="-", width=2, command=self.remove_table)
        self.btn_sub_table.pack(side='left')

        self.entry_tables_var = tk.StringVar(value="0")
        self.entry_tables = tk.Entry(self.frame_tables, textvariable=self.entry_tables_var, width=4, justify="center")
        self.entry_tables.pack(side='left', padx=2)
        self.entry_tables.bind('<Return>', self.apply_target_tables)

        self.btn_add_table = tk.Button(self.frame_tables, text="+", width=2, command=self.add_table)
        self.btn_add_table.pack(side='left')

        # 2. Trainers Control Sub-frame
        self.frame_trainers = tk.Frame(self.control_frame)
        self.frame_trainers.pack(side='left', padx=10)
        tk.Label(self.frame_trainers, text="Trainers:").pack(side='left')

        self.btn_sub_trainer = tk.Button(self.frame_trainers, text="-", width=2, command=self.remove_trainer)
        self.btn_sub_trainer.pack(side='left')

        self.entry_trainers_var = tk.StringVar(value="0")
        self.entry_trainers = tk.Entry(self.frame_trainers, textvariable=self.entry_trainers_var, width=4,
                                       justify="center")
        self.entry_trainers.pack(side='left', padx=2)
        self.entry_trainers.bind('<Return>', self.apply_target_trainers)

        self.btn_add_trainer = tk.Button(self.frame_trainers, text="+", width=2, command=self.add_trainer)
        self.btn_add_trainer.pack(side='left')

        # 3. Add a visual separator
        ttk.Separator(self.control_frame, orient='vertical').pack(side='left', fill='y', padx=10)

        # 4. Players Status Sub-frame (Moved to the right of the controls, larger fonts)
        self.frame_players = tk.Frame(self.control_frame)
        self.frame_players.pack(side='left', padx=10)

        self.lbl_players_total = tk.Label(self.frame_players, text="Total: 0", font=("Arial", 14, "bold"))
        self.lbl_players_total.pack(side='left', padx=4)

        self.lbl_players_waiting = tk.Label(self.frame_players, text="Wait: 0", font=("Arial", 14))
        self.lbl_players_waiting.pack(side='left', padx=4)

        self.lbl_players_playing = tk.Label(self.frame_players, text="Play: 0", font=("Arial", 14))
        self.lbl_players_playing.pack(side='left', padx=4)

        self.lbl_players_training = tk.Label(self.frame_players, text="Train: 0", font=("Arial", 14))
        self.lbl_players_training.pack(side='left', padx=4)

        # 5. Reset Button
        self.btn_reset = tk.Button(self.control_frame, text="Reset Defaults", command=self.reset_defaults, bg="#ffcccc",
                                   highlightbackground="#ffcccc")
        self.btn_reset.pack(side='right', padx=10)

        # Start the update loop
        self.refresh_data()
        self.root.mainloop()

    # --- UI Polish Methods ---
    def update_button_colors(self):
        # highlight background is required to color buttons on macOS, bg is for Windows/Linux

        # Tables
        if self.current_target_tables > self.actual_tables:
            self.btn_add_table.config(bg="#90ee90", highlightbackground="#90ee90")  # Light Green
            self.btn_sub_table.config(bg=self.default_btn_bg, highlightbackground=self.default_btn_hl)
        elif self.current_target_tables < self.actual_tables:
            self.btn_add_table.config(bg=self.default_btn_bg, highlightbackground=self.default_btn_hl)
            self.btn_sub_table.config(bg="#ff9999", highlightbackground="#ff9999")  # Light Red
        else:
            self.btn_add_table.config(bg=self.default_btn_bg, highlightbackground=self.default_btn_hl)
            self.btn_sub_table.config(bg=self.default_btn_bg, highlightbackground=self.default_btn_hl)

        # Trainers
        if self.current_target_trainers > self.actual_trainers:
            self.btn_add_trainer.config(bg="#90ee90", highlightbackground="#90ee90")
            self.btn_sub_trainer.config(bg=self.default_btn_bg, highlightbackground=self.default_btn_hl)
        elif self.current_target_trainers < self.actual_trainers:
            self.btn_add_trainer.config(bg=self.default_btn_bg, highlightbackground=self.default_btn_hl)
            self.btn_sub_trainer.config(bg="#ff9999", highlightbackground="#ff9999")
        else:
            self.btn_add_trainer.config(bg=self.default_btn_bg, highlightbackground=self.default_btn_hl)
            self.btn_sub_trainer.config(bg=self.default_btn_bg, highlightbackground=self.default_btn_hl)

    # --- Button Callbacks and Apply Functions ---
    def apply_target_tables(self, event=None):
        try:
            self.current_target_tables = max(0, int(self.entry_tables_var.get()))
            self.actor.set_target_tables.remote(target=self.current_target_tables)
            self.root.focus_set()
            self.update_button_colors()
            self.entry_tables_var.set(str(self.actual_tables))  # Snap text box back to actual
        except ValueError:
            pass

    def apply_target_trainers(self, event=None):
        try:
            self.current_target_trainers = max(0, int(self.entry_trainers_var.get()))
            self.actor.set_target_trainers.remote(target=self.current_target_trainers)
            self.root.focus_set()
            self.update_button_colors()
            self.entry_trainers_var.set(str(self.actual_trainers))
        except ValueError:
            pass

    def add_table(self):
        self.current_target_tables += 1
        self.actor.set_target_tables.remote(target=self.current_target_tables)
        self.update_button_colors()

    def remove_table(self):
        self.current_target_tables = max(0, self.current_target_tables - 1)
        self.actor.set_target_tables.remote(target=self.current_target_tables)
        self.update_button_colors()

    def add_trainer(self):
        self.current_target_trainers += 1
        self.actor.set_target_trainers.remote(target=self.current_target_trainers)
        self.update_button_colors()

    def remove_trainer(self):
        self.current_target_trainers = max(0, self.current_target_trainers - 1)
        self.actor.set_target_trainers.remote(target=self.current_target_trainers)
        self.update_button_colors()

    def reset_defaults(self):
        self.actor.reset_to_defaults.remote()
        self.root.focus_set()
        # The next refresh_data() tick will pull the new targets and update colors automatically

    def refresh_data(self):
        try:
            data = ray.get(self.actor.get_leaderboard_stats.remote())

            for item in self.tree.get_children():
                self.tree.delete(item)

            for i, p in enumerate(data['all_time'], 1):
                self.tree.insert("", "end", values=(
                    i, p['id'], p['game_count'], f"{p['total']:.2f}", f"{p['recent_avg']:.2f}",
                    p.get('status', 'Waiting')
                ))

            # self.lbl_players.config(text=f"Players: {data.get('num_players', 0)}")
            # Update player status counters
            self.lbl_players_total.config(text=f"Total: {data.get('num_players', 0)}")
            self.lbl_players_waiting.config(text=f"Wait: {data.get('num_waiting', 0)}")
            self.lbl_players_playing.config(text=f"Play: {data.get('num_playing', 0)}")
            self.lbl_players_training.config(text=f"Train: {data.get('num_training', 0)}")

            # Update state with backend data
            self.actual_tables = data.get('num_tables', 0)
            self.actual_trainers = data.get('num_trainers', 0)
            self.current_target_tables = data.get('target_tables', self.actual_tables)
            self.current_target_trainers = data.get('target_trainers', self.actual_trainers)

            # ONLY update text boxes to the ACTUAL number if user isn't typing
            if self.root.focus_get() != self.entry_tables:
                self.entry_tables_var.set(str(self.actual_tables))

            if self.root.focus_get() != self.entry_trainers:
                self.entry_trainers_var.set(str(self.actual_trainers))

            self.update_button_colors()

            if data.get('is_done'):
                self.label.config(text=f"Final Standings (Avg Games: {data['avg_games']:.1f}) - FINISHED")
                self.root.after(1000, self.root.destroy)
                return
            else:
                self.label.config(text=f"Live Standings (Avg Games: {data['avg_games']:.1f})")

        except Exception as e:
            print(f"Waiting for actor... {e}")
            self.root.after(1000, self.root.destroy)
            return

        self.root.after(1000, self.refresh_data)


if __name__ == "__main__":
    print("Connecting to Ray cluster inside Docker...")
    ray.init(address="ray://localhost:10001", namespace="casino")
    try:
        leaderboard_actor = ray.get_actor("GlobalLeaderboard")
        print("Successfully connected to the Leaderboard!")
        gui = LeaderboardGUI(leaderboard_actor)
    except Exception as e:
        print(e)
        print("Could not find the LeaderboardActor. Is the Casino running in Docker?")
    finally:
        ray.shutdown()