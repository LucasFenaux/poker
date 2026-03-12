import tkinter as tk
from tkinter import ttk
import ray


class LeaderboardGUI:
    def __init__(self, actor_handle):
        self.actor = actor_handle
        self.root = tk.Tk()
        self.root.title("Casino Live Leaderboard")
        self.root.geometry("500x400")

        # UI Elements
        self.label = tk.Label(self.root, text="Live Standings", font=("Arial", 16, "bold"))
        self.label.pack(pady=10)

        # Create a Treeview (Table)
        self.tree = ttk.Treeview(self.root, columns=("Rank", "ID", "Total", "Recent Avg"), show='headings')
        self.tree.heading("Rank", text="Rank")
        self.tree.heading("ID", text="Player ID")
        self.tree.heading("Total", text="All-Time")
        self.tree.heading("Recent Avg", text="Last 10 Avg")
        self.tree.column("Rank", width=50)
        self.tree.pack(expand=True, fill='both', padx=10, pady=10)

        # Start the update loop
        self.refresh_data()
        self.root.mainloop()

    def refresh_data(self):
        try:
            # Non-blocking request to the Ray Actor
            # We use ray.get here; since it's a small dict, it's fast.
            data = ray.get(self.actor.get_leaderboard_stats.remote())

            # Clear old entries
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Insert new entries (Top 10)
            for i, p in enumerate(data['all_time'], 1):
                self.tree.insert("", "end", values=(i, p['id'], f"{p['total']:.2f}", f"{p['recent_avg']:.2f}"))

            if data.get('is_done'):
                self.label.config(text=f"Final Standings (Games: {data['game_count']}) - FINISHED")
                return
            else:
                self.label.config(text=f"Live Standings (Games: {data['game_count']})")

        except Exception as e:
            print(f"Waiting for actor... {e}")

        # Schedule next update in 2000ms (2 seconds)
        self.root.after(2000, self.refresh_data)

# To run it:
# lb_actor = LeaderboardActor.remote(player_ids)
# ... start your casino logic ...
# gui = LeaderboardGUI(lb_actor)