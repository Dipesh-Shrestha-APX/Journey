from manim import *

class EigenvectorExplaination(Scene):
    def construct(self):
        # Title
        title = Text("Eigenvectors and Transformations", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Move title to corner
        self.play(
            title.animate.scale(0.6).to_corner(UL),
            run_time=1
        )
        
        # Create coordinate plane with grid
        plane = NumberPlane(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 2,
                "stroke_opacity": 0.6
            }
        )
        
        self.play(Create(plane), run_time=2)
        self.wait(0.5)
        
        # Explanation text
        explanation1 = Text(
            "Let's see how a matrix transforms space",
            font_size=30
        ).to_edge(DOWN)
        self.play(Write(explanation1))
        self.wait(1.5)
        
        # Create some regular vectors
        vector1 = Arrow(
            plane.c2p(0, 0),
            plane.c2p(1, 1),
            buff=0,
            color=YELLOW
        )
        vector2 = Arrow(
            plane.c2p(0, 0),
            plane.c2p(-1, 0.5),
            buff=0,
            color=GREEN
        )
        
        self.play(
            Create(vector1),
            Create(vector2)
        )
        self.wait(1)
        
        # Define transformation matrix
        # This matrix has eigenvectors along specific directions
        # Matrix: [[2, 1], [1, 2]]
        # Eigenvectors: [1, 1] with eigenvalue 3, and [1, -1] with eigenvalue 1
        
        self.play(FadeOut(explanation1))
        explanation2 = Text(
            "Applying transformation: [[2, 1], [1, 2]]",
            font_size=28
        ).to_edge(DOWN)
        self.play(Write(explanation2))
        self.wait(1)
        
        # Apply transformation
        matrix = [[2, 1], [1, 2]]
        
        # Transform vectors
        new_vector1 = Arrow(
            plane.c2p(0, 0),
            plane.c2p(
                matrix[0][0] * 1 + matrix[0][1] * 1,
                matrix[1][0] * 1 + matrix[1][1] * 1
            ),
            buff=0,
            color=YELLOW
        )
        
        new_vector2 = Arrow(
            plane.c2p(0, 0),
            plane.c2p(
                matrix[0][0] * (-1) + matrix[0][1] * 0.5,
                matrix[1][0] * (-1) + matrix[1][1] * 0.5
            ),
            buff=0,
            color=GREEN
        )
        
        self.play(
            plane.animate.apply_matrix(matrix),
            Transform(vector1, new_vector1),
            Transform(vector2, new_vector2),
            run_time=3
        )
        self.wait(1.5)
        
        # Reset and show eigenvectors
        self.play(
            FadeOut(vector1),
            FadeOut(vector2),
            FadeOut(plane),
            FadeOut(explanation2)
        )
        
        # Create fresh plane
        plane2 = NumberPlane(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 2,
                "stroke_opacity": 0.6
            }
        )
        self.play(Create(plane2), run_time=1)
        
        explanation3 = Text(
            "But some special vectors only scale!",
            font_size=32
        ).to_edge(DOWN)
        self.play(Write(explanation3))
        self.wait(1)
        
        # Show eigenvectors
        # Eigenvector 1: [1, 1] direction (eigenvalue 3)
        eigen_vec1 = Arrow(
            plane2.c2p(0, 0),
            plane2.c2p(1.5, 1.5),
            buff=0,
            color=RED,
            stroke_width=6
        )
        eigen_label1 = Text("v₁", color=RED, font_size=36)  # <-- Replaced MathTex
        eigen_label1.next_to(eigen_vec1.get_end(), UR)
        
        # Eigenvector 2: [1, -1] direction (eigenvalue 1)
        eigen_vec2 = Arrow(
            plane2.c2p(0, 0),
            plane2.c2p(1.5, -1.5),
            buff=0,
            color=PURPLE,
            stroke_width=6
        )
        eigen_label2 = Text("v₂", color=PURPLE, font_size=36)  # <-- Replaced MathTex
        eigen_label2.next_to(eigen_vec2.get_end(), DR)
        
        self.play(
            Create(eigen_vec1),
            Write(eigen_label1),
            Create(eigen_vec2),
            Write(eigen_label2)
        )
        self.wait(2)
        
        self.play(FadeOut(explanation3))
        explanation4 = Text(
            "These are EIGENVECTORS - they only stretch!",
            font_size=32
        ).to_edge(DOWN)
        self.play(Write(explanation4))
        self.wait(1)
        
        # Transform the eigenvectors
        new_eigen_vec1 = Arrow(
            plane2.c2p(0, 0),
            plane2.c2p(
                matrix[0][0] * 1.5 + matrix[0][1] * 1.5,
                matrix[1][0] * 1.5 + matrix[1][1] * 1.5
            ),
            buff=0,
            color=RED,
            stroke_width=6
        )
        
        new_eigen_vec2 = Arrow(
            plane2.c2p(0, 0),
            plane2.c2p(
                matrix[0][0] * 1.5 + matrix[0][1] * (-1.5),
                matrix[1][0] * 1.5 + matrix[1][1] * (-1.5)
            ),
            buff=0,
            color=PURPLE,
            stroke_width=6
        )
        
        new_label1 = Text("v₁", color=RED, font_size=36)  # <-- Replaced MathTex
        new_label1.next_to(new_eigen_vec1.get_end(), UR)
        
        new_label2 = Text("v₂", color=PURPLE, font_size=36)  # <-- Replaced MathTex
        new_label2.next_to(new_eigen_vec2.get_end(), DR)
        
        self.play(
            plane2.animate.apply_matrix(matrix),
            Transform(eigen_vec1, new_eigen_vec1),
            Transform(eigen_label1, new_label1),
            Transform(eigen_vec2, new_eigen_vec2),
            Transform(eigen_label2, new_label2),
            run_time=3
        )
        self.wait(2)
        
        # Final explanation
        self.play(FadeOut(explanation4))
        explanation5 = Text(
            "Eigenvectors maintain their direction!",
            font_size=32,
            color=YELLOW
        ).to_edge(DOWN)
        self.play(Write(explanation5))
        self.wait(2)
        
        # Fade out everything
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1.5
        )
        self.wait(0.5)
