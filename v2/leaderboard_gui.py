import tkinter as tk
from tkinter import ttk
import ray


class LeaderboardGUI:
    def __init__(self, actor_handle):
        self.actor = actor_handle
        self.root = tk.Tk()
        self.root.title("Casino Live Leaderboard")
        self.root.geometry("600x400")  # Made it slightly wider to fit the new column

        # UI Elements
        self.label = tk.Label(self.root, text="Live Standings", font=("Arial", 16, "bold"))
        self.label.pack(pady=10)

        # Added "Games" to the columns list
        self.tree = ttk.Treeview(self.root, columns=("Rank", "ID", "Games", "Total", "Recent Avg"), show='headings')
        self.tree.heading("Rank", text="Rank")
        self.tree.heading("ID", text="Player ID")
        self.tree.heading("Games", text="Games")  # New heading
        self.tree.heading("Total", text="All-Time")
        self.tree.heading("Recent Avg", text="Last 10 Avg")

        self.tree.column("Rank", width=50)
        self.tree.column("Games", width=60)  # Set width for new column
        self.tree.pack(expand=True, fill='both', padx=10, pady=10)

        # Start the update loop
        self.refresh_data()
        self.root.mainloop()

    def refresh_data(self):
        try:
            # Fetch data from the actor
            data = ray.get(self.actor.get_leaderboard_stats.remote())

            # Clear old entries
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Insert new entries including the game_count
            for i, p in enumerate(data['all_time'], 1):
                self.tree.insert("", "end", values=(
                    i,
                    p['id'],
                    p['game_count'],  # Insert the individual player's game count
                    f"{p['total']:.2f}",
                    f"{p['recent_avg']:.2f}"
                ))

            # Update the label using the overall total_games
            if data.get('is_done'):
                self.label.config(text=f"Final Standings (Total Games: {data['total_games']}) - FINISHED")
                self.root.after(1000, self.root.destroy)
                return
            else:
                self.label.config(text=f"Live Standings (Total Games: {data['total_games']})")

        except Exception as e:
            print(f"Waiting for actor... {e}")

        # Schedule next update in 2000ms (2 seconds)
        self.root.after(2000, self.refresh_data)