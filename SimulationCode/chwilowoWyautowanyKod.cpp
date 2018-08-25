 int tmp = 6*prevTop;
            
            bool res = true;
            if(!(pos[thid*3] >= b[tmp] && pos[thid*3] <= b[tmp + 1]) && res) res = false;
            if(!(pos[thid*3 + 2] >= b[tmp] && pos[thid*3 + 1] <= b[tmp + 3]) && res) res = false;
            if(!(pos[thid*3 + 4] >= b[tmp] && pos[thid*3 + 2] <= b[tmp + 5]) && res) res = false;
            if(thid == 7)
             printf("a tu\n", top);
                
            if(res)
            {
                printf("jeszcze raz\n");
                //rekurencja pomijajac ratio
                if(nextChild==numberOfChilds) {
                    continue;
                }
                    printf("tak tu\n");
                stack[++top] = idx;
                child[top] = nextChild++;
                /*
                for(int j=0; j<6; j++)
                    stack[6*top + j] = stack[6*prevTop + j];
                */
                // if(octree[idx].position >= 0) // tutaj niespelnione trywialnie
                /*stack[++top] = octree[idx].children[nextChild-1];
                child[top] = 0;
                for(int i=0; i<3; i++) {
                    if(multipliers[nextChild - 1][i]) {
                        b[top*6 + 2*i] = b[prevTop*6 + 2*i];
                        b[top*6 + 2*i + 1] = b[prevTop*6 + 2*i] + (b[prevTop*6 + 2*i + 1] - b[prevTop*6 + 2*i])/2;
                    } else {
                        b[top*6 + 2*i] = b[prevTop*6 + 2*i] + (b[prevTop*6 + 2*i + 1] - b[prevTop*6 + 2*i])/2;
                        b[top*6 + 2*i + 1] = b[prevTop*6 + 2*i + 1];
                    }
                }*/
                continue;
            }
             printf("kurwa\n");
            // jakos wywalic i!=j

