from digitRecognizer import *
import pygame
import sys, random
pygame.init()

(WIDTH,HEIGHT) = (560,560)#28 * 20
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Drawing Interface: Digit Recognizer -- Justin Stitt")
clock = pygame.time.Clock()
fps = 120
background_color = (0,0,0)

font = pygame.font.Font('freesansbold.ttf', 20)
text_color = (26, 188, 237)
p_text = font.render('Prediction: Draw Something!!',True,text_color)

lmb = False

cells = []

class Cell:
    def __init__(self, pos, color):
        self.pos = pos
        self.color = color
        self.size = WIDTH//28#28 is size of MNIST data set (28x28)
        self.luma = 0
    def update(self):
        self.render()
    def render(self):
        pygame.draw.rect(screen,self.color,(*self.pos,self.size,self.size))
    def change_color(self,new_color):
        self.color = new_color
        self.luma = 0.2989 * self.color[0] + 0.5870 * self.color[1] + 0.1140 * self.color[2]
        #normalize luma
        self.luma /= 255.

def update():
    global p_text
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                to_predict_img = convert_grid_to_image(cells)
                #make our prediction
                prediction = get_prediction(to_predict_img)
                best_guess = np.argmax(prediction)
                p_text = font.render('Prediction: {}'.format(best_guess),True,text_color)
                print('prediction: {}'.format(best_guess))
            elif event.key == pygame.K_DOWN:
                for row in cells:
                    for cell in row:
                        cell.change_color((0,0,0))#reset!
                        p_text = font.render('Prediction: Draw Something!',True,text_color)

    pressed = pygame.mouse.get_pressed()
    if(pressed == (1,0,0)):#lmb pushed
        lmb = True
    else:
        lmb = False
    if(lmb):#is lmb currently held down?
        mpos = pygame.mouse.get_pos()
        #print(translate_pos_to_coord(mpos))
        cell_pos = translate_pos_to_coord(mpos)
        cells[cell_pos[0]][cell_pos[1]].change_color((255,255,255))

    #obj updates
    for row in cells:
        for cell in row:
            cell.update()

def setup_grid():
    global cells
    for x in range(28):
        cells.append([])
        for y in range(28):
            #cells.append(Cell([x * 20,y*20],(255,random.randint(0,255),random.randint(0,255))))
            cells[x].append(Cell([y*20,x*20],(0,0,0)))


def translate_pos_to_coord(pos):
    r = pos[1]//20
    c = pos[0]//20
    return [r,c]


def render():
    pass

setup_grid()


while True:
    screen.fill(background_color)

    update()
    render()
    screen.blit(p_text,(WIDTH//12,HEIGHT - 45))

    pygame.display.flip()
    clock.tick(fps)
