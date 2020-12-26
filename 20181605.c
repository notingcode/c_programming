#define NAMELENGTH 70
#define TCLENGTH 20 // 티켓과 객실 정보를 담을 char string의 길이
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

enum Sex{Male=0, Female};
enum PClass{First=0, Second, Third};

typedef struct{
    int passengerId;
    int survival;
    char* name;
    enum Sex sex;
    float age;
    int sibsp;
    int parch;
    char* ticket;
    char* cabin;
    char* embarked;
    double fare;
    enum PClass pclass;
}SUVData; // 탑승자의 정보를 담은 구조체

typedef struct{
    int survived;
    int death;
    float ratio;
}SDRatio; // 각 카테고리의 생존자 수/사망자 수/생존율 통계를 담는 구조체

typedef struct{
    int ageStart;
    int ageEnd;
}AgeRange; // 나이 별로 분류 시 [최소, 최대]를 담는 구조체

typedef struct{
    SDRatio male;
    SDRatio female;
}SexStats; // 성별에 따라 카테고리를 분류

typedef struct{
    AgeRange range;
    SDRatio ageStep;
}AgeStats; // 나이에 따라 카테고리를 분류하고 각 분류의 나이의 범위와 통계를 담는 구조체

int ReadData(SUVData** dataset, char* filename); // 데이터셋 포인터와 파일 이름을 전달받아 파일 내용을 데이터셋 포인터에 저장하는 함수
void StatsBySex(SUVData** dataset, int N); // 성별(남, 여) 통계를 내는 함수
void StatsByFamily(SUVData** dataset, int N); // 가족의 수 별로 통계를 내는 함수
void StatsByAge(int step, SUVData** dataset, int N); // 나이를 step개의 범위로 나누어 통계를 내는 함수
void CalCorrelationMatrix(SUVData** dataset, int N); // 상관 계수 행렬을 출력하는 함수
void KNN(SUVData** dataset, int N, int K); // KNN 알고리즘을 수행하는 함수
void KFoldValidation(SUVData** dataset, int N, int nFold, int K); // 데이터셋을 nFold개로 분할하여 KNN 알고리즘을 수행하는 함수
char* numToPerString(float number); // float타입의 비율(%)을 출력을 위해 char string으로 변경하여 리턴하는 함수
double normalize(double num, double min_n, double max_n); // double타입의 수를 받아 정규화하고 리턴하는 함수 
float EuclidDist(double* X, double* X_prime, int num_feature); // 벡터 X와 X_prime을 받아 두 벡터의 유클리드 거리를 구하고 리턴하는 함수
void StatsByEmbarked(SUVData** dataset, int N); // 탑승했던 항구에 따라 생존자의 통계를 내는 함수

int main(int argc, char* argv[]){
    int menu_choice, passNum, step, K, nFold;
    SUVData** Passengers;

    Passengers = (SUVData**)malloc(sizeof(SUVData*));

    passNum = ReadData(Passengers, argv[1]);

    printf("총 탑승객 수: %d\n\n", passNum);

    if(passNum < 1){
        return -1;
    }
    else{
        // 메뉴 출력
        printf("[타이타닉 생존자 데이터 분석 - 메뉴를 선택하시오]\n");
        printf("0. 출발 항구 기준 생존자 수 비율\n");
        printf("1. 성별 기준 생존자 수 비율\n");
        printf("2. 가족 관계 별 생존자 수 비율\n");
        printf("3. 나이대 별 생존자 수 비율\n");
        printf("4. Correlation Matrix 게산하기\n");
        printf("5. 최근접 이웃 알고리즘으로 생존자 수 예측하기\n");
        printf("6. K-Fold Vlidation\n");
        printf("7. 프로그램 종료\n\n");

        do{
            printf("[명령 입력] > ");
            scanf("%d", &menu_choice); 
            // 사용자가 입력한 명령에 따라 함수를 실행한다. 만약 사용자가 0~7외의 값을 입력하였을 경우 다시 입력을 받는다.
            switch(menu_choice){
                case 0:
                    StatsByEmbarked(Passengers, passNum);
                    break;
                case 1:
                    StatsBySex(Passengers, passNum);
                    break;
                case 2:
                    StatsByFamily(Passengers, passNum);
                    break;
                case 3:
                    printf("등급의 개수를 입력하세요: ");
                    scanf("%d", &step);
                    StatsByAge(step, Passengers, passNum);
                    break;
                case 4:
                    printf("> CalCorrelationMatrix()\n");
                    CalCorrelationMatrix(Passengers, passNum);
                    break;
                case 5:
                    printf("> KNN()\n");
                    printf("Input K: ");
                    scanf("%d", &K);
                    KNN(Passengers, passNum, K);
                    break;
                case 6:
                    printf("Input nFolds and K: ");
                    scanf("%d %d", &nFold, &K);
                    KFoldValidation(Passengers, passNum, nFold, K);
                    break;
                case 7:
                    printf("\n프로그램 종료됨\n");
                    break;
                default:
                    printf("해당 메뉴 옵션 없음\n");
                    break;
            }
        }while(menu_choice != 7);
    }

    return 0;
}

// filename(파일이름)을 열어 파일에 저장된 데이터셋을 읽는다
int ReadData(SUVData** dataset, char* filename){
    FILE* fp;
    char *line, *temp_sex; // 파일의 각 줄을 저장할 line과 성별("male", "female")을 저장할 temp_sex
    int n, memory_size, temp_pclass;

    fp = fopen(filename, "r");

    if(!fp){
        printf("Error: file not found\n");
        return -1;
    }
    else{
        *dataset = (SUVData*)calloc(1, sizeof(SUVData));
        (*dataset+0)->name = (char*)malloc(NAMELENGTH * sizeof(char));
        (*dataset+0)->ticket = (char*)malloc(TCLENGTH * sizeof(char));
        (*dataset+0)->cabin = (char*)malloc(TCLENGTH * sizeof(char));
        (*dataset+0)->embarked = (char*)malloc(sizeof(char));

        line = (char*)malloc(1024 * sizeof(char));
        temp_sex = (char*)malloc(7 * sizeof(char));

        memory_size = 1;
        n = 0;
        while(fgets(line, 1024, fp)){
            //
            if(sscanf(line, "%d,%d,%d,%[^,],%[^,],%f,%d,%d,%[^,],%lf,%[^,],%c", &((*dataset+n)->passengerId),&((*dataset+n)->survival), &temp_pclass, ((*dataset+n)->name), temp_sex, &((*dataset+n)->age), &((*dataset+n)->sibsp), &((*dataset+n)->parch), (*dataset+n)->ticket, &((*dataset+n)->fare), (*dataset+n)->cabin, (*dataset+n)->embarked)){
                // temp_sex에 저장된 string에 따라 male=0, female=0로 변환
                if(strcmp(temp_sex,"male") == 0){
                    (*dataset+n)->sex = Male;
                }
                else{
                    (*dataset+n)->sex = Female;
                }

                switch(temp_pclass){
                    case 1:
                        (*dataset+n)->pclass = First;
                        break;
                    case 2:
                        (*dataset+n)->pclass = Second;
                        break;
                    default:
                        (*dataset+n)->pclass = Third;
                }
                ++n;
                // 데이터셋의 실제 크기가 지정된 메모리 크기보다 클 경우 메모리의 크기를 2배로 늘린다
                if(n == memory_size){
                    memory_size *= 2;
                    *dataset = (SUVData*)realloc(*dataset, memory_size * sizeof(SUVData));
                    if(!(*dataset)){
                        printf("Error: realloc failed\n");
                        return -1;
                    }
                }
                (*dataset+n)->name = (char*)malloc(NAMELENGTH * sizeof(char));
                (*dataset+n)->ticket = (char*)malloc(TCLENGTH * sizeof(char));
                (*dataset+n)->cabin = (char*)malloc(TCLENGTH * sizeof(char));
                (*dataset+n)->embarked = (char*)malloc(sizeof(char));
            }
            else{
                printf("WARNING: passenger data skipped at line %d\n", n+1);
            }
        }
        free(line);
        free(temp_sex);
    }

    return n;
}

// Male과 Female별로 생존자/사망자/생존율 통계를 내는 함수
void StatsBySex(SUVData** dataset, int N){
    SexStats *stats = (SexStats*)calloc(1, sizeof(SexStats));
    int sdSum;
    char *f_ratio, *m_ratio;

    for(int i = 0; i < N; ++i){
        if((*dataset+i)->sex == Male){
            switch((*dataset+i)->survival){
                case 0:
                    ++(stats->male.death);
                    break;
                default:
                    ++(stats->male.survived);
                    break;
            }
        }
        else{
            switch((*dataset+i)->survival){
                case 0:
                    ++(stats->female.death);
                    break;
                default:
                    ++(stats->female.survived);
                    break;
            }
        }
    }

    // 성별마다 생존율을 계산
    // 만약 sdSum(생존자+사망자)가 0이라면 해당 성별의 데이터가 존재하지 않으므로 ratio를 0으로 입력
    if((sdSum = (stats->female.survived + stats->female.death)) == 0){
        stats->female.ratio = 0;
    }
    else{
        stats->female.ratio = (float)stats->female.survived/sdSum * 100.0;
    }

    if((sdSum = (stats->male.survived + stats->male.death)) == 0){
        stats->male.ratio = 0;
    }
    else{
        stats->male.ratio = (float)stats->male.survived/sdSum * 100.0;
    }

    // 출력 형식을 맞추기 위해 생존율을 소수점 1자리까지 char string으로 변환
    f_ratio = numToPerString(stats->female.ratio);
    m_ratio = numToPerString(stats->male.ratio);

    // 계산된 통계를 출력한다
    printf("\n%-12s%s\n", " ", "Sex");
    printf("%-12s%-12s%s\n", " ", "Female", "Male");
    printf("%-12s%-12d%d\n", "Survived", stats->female.survived, stats->male.survived);
    printf("%-12s%-12d%d\n", "Dead", stats->female.death, stats->male.death);
    printf("%-12s%-12s%s\n\n", "Ratio", f_ratio, m_ratio);

    free(stats);
    free(f_ratio);
    free(m_ratio);

    return;
}

// 가족의 수(parch+sibsp)에 따라 데이터를 분류하여 각 카테고리의 생존자/사망자/생존율 통계를 내는 함수
void StatsByFamily(SUVData** dataset, int N){
    int curr_fsize, num_fsize, sdSum;
    char* str_ratio;

    num_fsize = 9;

    SDRatio *fSize = (SDRatio*)calloc(num_fsize, sizeof(SDRatio));

    for(int i = 0; i < N; ++i){
        curr_fsize = ((*dataset+i)->sibsp)+((*dataset+i)->parch);
        // 가족의 수가 7명을 넘으면 마지막 카테고리에 모아서 분류
        if(curr_fsize > 7){
            ((*dataset+i)->survival == 0) ? ++((fSize+8)->death) : ++((fSize+8)->survived);
        }
        // 그 외의 가족의 수(0~7)에 따라 분류
        else{
            ((*dataset+i)->survival == 0) ? ++((fSize+curr_fsize)->death) : ++((fSize+curr_fsize)->survived);
        }
    }

    // 통계 출력
    printf("%-15s%s\n", " ", "Family Size");
    printf("%-15s%-7d%-7d%-7d%-7d%-7d%-7d%-7d%-7d%s\n", " ", 0,1,2,3,4,5,6,7, ">7");
    printf("%-15s", "Survived");
    // 생존자의 수 출력
    for(int i = 0; i < num_fsize; ++i){
        printf("%-7d", (fSize+i)->survived);
    }
    // 사망자의 수 출력
    printf("\n%-15s", "Dead");
    for(int i = 0; i < num_fsize; ++i){
        printf("%-7d", (fSize+i)->death);
    }
    // 생존율 출력
    printf("\n%-15s", "Ratio");
    for(int i = 0; i < num_fsize; ++i){
        sdSum = ((fSize+i)->death) + ((fSize+i)->survived);
        if(sdSum == 0){
            (fSize+i)->ratio = 0.0;
        }
        else{
            (fSize+i)->ratio = (float)((fSize+i)->survived)/sdSum * 100;
        }
        str_ratio = numToPerString((fSize+i)->ratio);
        printf("%-7s", str_ratio);
        free(str_ratio);
    }
    printf("\n\n");

    free(fSize);

    return;
}

// 나이(0~100세)를 step개의 구간으로 나누어 각 구간마다 생존자/사망자/생존율 통계를 내는 함수 
void StatsByAge(int step, SUVData** dataset, int N){
    int curr_age, interval, sdSum;
    char* str_ratio;
    AgeStats *stats = (AgeStats*)calloc(step, sizeof(AgeStats));

    if(step > 100){
        step = 10;
        printf("Warning: step-size is greater than 100, the step-size value changed to %d\n", step);
    }

    interval = 100/step;

    for(int i = 0; i < N; ++i){
        curr_age = (*dataset+i)->age;
        for(int j = 0; j < step; ++j){
            // [ageStart, ageEnd]로 연령구간을 나누어 통계를 낸다
            ((stats+j)->range).ageStart = (interval*j); // interval을 기준으로 각 구간의 ageStart를 계산
            ((stats+j)->range).ageEnd = (interval*(j+1))+((1+100%step)*((j+1)/step))-1; // 100이 step의 배수가 아닌 경우에 대응하고 마지막 구간은 100을 포함해야 하기 때문에 ((1+100%step)*((j+1)/step))-1을 더한다
            if((curr_age >= ((stats+j)->range).ageStart) && (curr_age <= ((stats+j)->range).ageEnd)){
                (*dataset+i)->survival == 0 ? ++(((stats+j)->ageStep).death) : ++(((stats+j)->ageStep).survived);
            }
        }
    }

    // 통계를 출력
    printf("%-15s%s\n%-15s", " ", "Age Range", " ");
    for(int i = 0; i < step; ++i){
        printf("%3d~%-4d", (int)((stats+i)->range).ageStart, (int)((stats+i)->range).ageEnd);
    }
    printf("\n%-15s", "Survived");
    for(int i = 0; i < step; ++i){
        printf("%-8d", ((stats+i)->ageStep).survived);
    }
    printf("\n%-15s", "Dead");
    for(int i = 0; i < step; ++i){
        printf("%-8d", ((stats+i)->ageStep).death);
    }
    printf("\n%-15s", "Ratio");
    for(int i = 0; i < step; ++i){
        sdSum = ((stats+i)->ageStep).death + ((stats+i)->ageStep).survived;
        if(sdSum == 0){
            ((stats+i)->ageStep).ratio = 0;
        }
        else{
            (((stats+i)->ageStep).ratio) = (float)(((stats+i)->ageStep).survived)/sdSum * 100;
        }
        str_ratio = numToPerString(((stats+i)->ageStep).ratio);
        printf("%-8s", str_ratio);
        free(str_ratio);
    }

    printf("\n\n");

    free(stats);

    return;
}

// 상관 계수 행렬을 출력하는 함수
void CalCorrelationMatrix(SUVData** dataset, int N){
    char start_tag = 'a';
    int num_var = 7;
    float **data_table;
    double **corr_matrix, *means, *std;

    means = (double*)calloc(num_var, sizeof(double)); // 각 variable의 mean을 저장할 메모리
    std = (double*)calloc(num_var, sizeof(double)); // 각 variable의 standard deviation을 저장할 메모리

    data_table = (float**)malloc(num_var * sizeof(float*)); // Variable을 2차원 data_table에 임시로 저장하여 for loop을 통한 계산을 할 수 있게 한다
    corr_matrix = (double**)calloc(num_var, sizeof(double*)); // Correlation matrix
    for(int row = 0; row < num_var; ++row){
        *(data_table+row) = (float*)calloc(N, sizeof(float)); // 각 variable마다 N개의 데이터가 있다
        *(corr_matrix+row) = (double*)calloc(num_var, sizeof(double)); // Correlation matrix의 크기는 num_var * num_var
    }

    // 각 variable의 tag를 출력한다
    printf("[Define Variables]\n");
    printf("%-8s = a, %-8s = b, %-8s = c, %-8s = d, %-8s = e\n%-8s = f, %-8s = g\n\n", "Survival", "PCLass", "Sex", "Age", "Sibsp", "Parch", "Fare");

    for(int i = 0; i < N; ++i){
        *(*(data_table+0)+i) = (*dataset+i)->survival;
        *(*(data_table+1)+i) = (*dataset+i)->pclass;
        *(*(data_table+2)+i) = (*dataset+i)->sex;
        *(*(data_table+3)+i) = (*dataset+i)->age;
        *(*(data_table+4)+i) = (*dataset+i)->sibsp;
        *(*(data_table+5)+i) = (*dataset+i)->parch;
        *(*(data_table+6)+i) = (*dataset+i)->fare;
    }

    // 각 variable의 mean value를 계산하고 저장
    for(int i = 0; i < num_var; ++i){
        for(int j = 0; j < N; ++j){
            *(means+i) += *(*(data_table+i)+j);
        }
        *(means+i)/=N;
    }

    // 각 variable의 standard deviation을 계산하고 저장
    for(int i = 0; i < num_var; ++i){
        for(int n = 0; n < N; ++n){
            *(std+i) += pow((*(*(data_table+i)+n)) - *(means+i), 2.0);
        }
        *(std+i) /= (N-1);
        *(std+i) = sqrt(*(std+i));
    }

    // Correlation matrix를 계산하고 저장
    for(int col = 0; col < num_var; ++col){
        // Correlation matrix의 대각선과 그 위의 값만 계산한다
        for(int row = 0; row < (col+1); ++row){
            for(int n = 0; n < N; ++n){
                *(*(corr_matrix+row)+col) += ((*(*(data_table+row)+n) - *(means+row)) * (*(*(data_table+col)+n) - *(means+col)));
            }
            *(*(corr_matrix+row)+col) /= ((N-1)*(*(std+row))*(*(std+col)));
            *(*(corr_matrix+col)+row) = *(*(corr_matrix+row)+col); // Correlation matrix는 대칭 행렬이기 때문에 (row_i, column_j)의 값이 (row_j, column_i)의 값과 같다
        }
    }

    // Correlation matrix를 출력한다
    printf("   ");
    for(char tag = 'a'; tag <= 'g'; ++tag){
        printf("%c      ", tag);
    }
    printf("\n");
    for(int i = 0; i < num_var; ++i){
        printf("%c", start_tag+i);
        for(int j = 0; j < num_var; ++j){
            printf("  ");
            if(*(*(corr_matrix+i)+j) > 0){
                printf("+%3.2f", *(*(corr_matrix+i)+j));
            }
            else{
                printf("%3.2f", *(*(corr_matrix+i)+j));
            }
        }
        printf("\n");
    }

    // corr_matrix와 나머지 메모리를 해제
    for(int row = 0; row < num_var; ++row){
        free(*(data_table+row));
        free(*(corr_matrix+row));
    }
    free(data_table);
    free(means);
    free(std);
    free(corr_matrix);

    printf("\n");

    return;
}

// KNN 알고리즘을 수행
void KNN(SUVData** dataset, int N, int K){
    typedef struct{
        float distance;
        int index;
    }Dist; // 

    int i, j, p, num_features, num_survival, num_death, curr_idx, num_correct, survival_state;
    int train_size, test_size, min_sibsp, max_sibsp, min_parch, max_parch, min_pclass, max_pclass;
    float min_age, max_age, score;
    double *X, *X_prime, min_fare, max_fare;

    Dist** dists;
    SUVData** P;
    SUVData** Q;

    num_features = 6;
    X = (double*)malloc(num_features * sizeof(double));
    X_prime = (double*)malloc(num_features * sizeof(double));

    // K가 1보다 작을 경우 실행
    if(K < 1){
        K = 1; // K가 가질 수 있는 최소값
        printf("Warning: K cannot be < 1, K value changed to %d\n", K);
    }

    // 전체 데이터의 첫 80%를 훈련 데이터로 마지막 20%를 테스트 데이터의 크기로 한다
    train_size = (N*4);
    if(train_size%5 != 0){
        train_size /= 5;
        ++train_size;
    }
    else{
        train_size /= 5;
    }

    // K가 훈련 데이터의 크기보다 클 경우 실행
    if(K > train_size){
        K = train_size; // K가 가질 수 있는 최대값
        printf("Warning: K-value is larger than train size\n> K-value modified to %d\n", K);
    }

    test_size = N-train_size;

    // 거리 계산 결과를 저장할 dists와 훈련데이터, 테스트 데이터들의 주소를 저장하는 P,Q 배열을 선언한다
    dists = (Dist**)malloc(train_size * sizeof(Dist*));
    Q = (SUVData**)malloc(train_size * sizeof(SUVData*));
    P = (SUVData**)malloc(test_size * sizeof(SUVData*));

    // 훈련데이터와 테스트 데이터에 dataset의 데이터의 주소를 저장한다
    for(j = 0; j < train_size; ++j){
        *(Q+j) = (*dataset+j);
        *(dists+j) = (Dist*)malloc(sizeof(Dist));
    }
    for(i = 0; i < test_size; ++i){
        *(P+i) = (*dataset+train_size+i);
    }

    // 각 항목의 최대값과 최소값을 찾는다. 우선 최대, 최소값을 첫번째 인자로 통일한다
    min_age = max_age = (*(Q))->age;
    min_fare = max_fare = (*(Q))->fare;
    min_parch = max_parch = (*(Q))->parch;
    min_sibsp = max_sibsp = (*(Q))->sibsp;
    min_pclass = max_pclass = (*(Q))->parch;
    // 최소값 찾기
    for(j = 1; j < train_size; ++j){
        if(min_age > (*(Q+j))->age){
            min_age = (*(Q+j))->age;
        }
        if(min_fare > (*(Q+j))->fare){
            min_fare = (*(Q+j))->fare;
        }
        if(min_parch > (*(Q+j))->parch){
            min_parch = (*(Q+j))->parch;
        }
        if(min_sibsp > (*(Q+j))->sibsp){
            min_sibsp = (*(Q+j))->sibsp;
        }
        if(min_pclass > (*(Q+j))->parch){
            min_pclass = (*(Q+j))->parch;
        }
    }
    // 최대값 찾기
    for(j = 1; j < train_size; ++j){
        if(max_age < (*(Q+j))->age){
            max_age = (*(Q+j))->age;
        }
        if(max_fare < (*(Q+j))->fare){
            max_fare = (*(Q+j))->fare;
        }
        if(max_parch < (*(Q+j))->parch){
            max_parch = (*(Q+j))->parch;
        }
        if(max_sibsp < (*(Q+j))->sibsp){
            max_sibsp = (*(Q+j))->sibsp;
        }
        if(max_pclass < (*(Q+j))->parch){
            max_pclass = (*(Q+j))->parch;
        }
    }

    p = 1;
    num_correct = 0;
    score = 0;
    for(i = 0; i < test_size; ++i){
        // 테스트 데이터의 생존 여부를 확인
        survival_state = (*(P+i))->survival;
        // 테스트 데이터의 각 항목을 X에 정규화하여 저장
        *(X) = normalize((*(P+i))->pclass, min_pclass, max_pclass);
        *(X+1) = (*(P+i))->sex;
        *(X+2) = normalize((*(P+i))->age, min_age, max_age);
        *(X+3) = normalize((*(P+i))->sibsp, min_sibsp, max_sibsp);
        *(X+4) = normalize((*(P+i))->parch, min_parch, max_parch);
        *(X+5) = normalize((*(P+i))->fare, min_fare, max_fare);
        for(j = 0; j < train_size; ++j){
            // 훈련 데이터의 각 항목을 X_prime에 정규화하여 저장
            *(X_prime) = normalize((*(Q+j))->pclass, min_pclass, max_pclass);
            *(X_prime+1) = (*(Q+j))->sex;
            *(X_prime+2) = normalize((*(Q+j))->age, min_age, max_age);
            *(X_prime+3) = normalize((*(Q+j))->sibsp, min_sibsp, max_sibsp);
            *(X_prime+4) = normalize((*(Q+j))->parch, min_parch, max_parch);
            *(X_prime+5) = normalize((*(Q+j))->fare, min_fare, max_fare);
            // 테스트 데이터와 훈련 데이터 사이의 거리를 계산 후 저장
            (*(dists+j))->distance = EuclidDist(X, X_prime, num_features);
            (*(dists+j))->index = j;
        }
        // 저장된 거리 데이터를 가까운 순서로 정렬
        for(j = 0; j < train_size-1; ++j){
            for(int w = 0; w < train_size-j-1; ++w){
                if(((*(dists+w))->distance) > ((*(dists+w+1))->distance)){
                    Dist *temp;
                    temp = *(dists+w);
                    *(dists+w) = *(dists+w+1);
                    *(dists+w+1) = temp;
                }
            }
        }
        // 가장 가까운 K번째 인자까지 생존자 비율을 확인
        num_survival = num_death = 0;
        for(int k = 0; k < K; ++k){
            curr_idx = (*(dists+k))->index;
            if((*(Q+curr_idx))->survival == 0){
                num_death++;
            }
            else{
                num_survival++;
            }
        }
        // 생존자와 사망자의 수가 같으면 K+1번째 인자까지 확인
        if(num_death == num_survival){
            curr_idx = (*(dists+K))->index;
            if((*(Q+curr_idx))->survival == 0){
                num_death++;
            }
            else{
                num_survival++;
            }
        }
        // 예측이 맞았을 경우 맞은 개수를 증가시킨다
        if(num_survival > num_death){
            if(survival_state == 1)
                ++num_correct;
        }
        else{
            if(survival_state == 0){
                ++num_correct;
            }
        }
        // 진행도가 20% 증가할 때마다 진행도를 출력한다
        if(((int)(((float)(i+1)/test_size)*100)) == (20*p)){
            printf("[--------- %d/%d %3d%% completed ---------]\n", i+1, test_size, (20*p));
            ++p;
        }
    }

    // 최종 정확도를 계산하고 출력한다
    score = (float)num_correct/test_size * 100;

    printf("\nPrediction Score: %.2f%%\n\n", score);

    for(j = 0; j < train_size; ++j){
        free(*(dists+j));
    }

    free(X);
    free(X_prime);
    free(dists);
    free(P);
    free(Q);

    return;
}

// 데이터셋을 nFold개로 분할하여 KNN 알고리즘을 수행하는 함수
void KFoldValidation(SUVData** dataset, int N, int nFold, int K){
    typedef struct{
        float distance;
        int index;
    }Dist;

    int i, j, p, num_features, num_survival, num_death, curr_idx, num_correct, survival_state;
    int train_size, test_size, min_sibsp, max_sibsp, min_parch, max_parch, min_pclass, max_pclass;
    float min_age, max_age, score;
    double *X, *X_prime, min_fare, max_fare;

    Dist** dists;
    SUVData** P;
    SUVData** Q;

    // 분할 구간의 수가 전체 데이터의 수보다 클 경우 실행
    if(nFold > N){
        nFold = N;// nFold가 가질 수 있는 최대값
        printf("Warning: nFold is larger than the dataset\n> nFold value changed to %d\n", N);
    }
    // 분할 구간 2개보다 작을 경우 실행
    if(nFold < 2){
        nFold = 2; // nFold가 가질 수 있는 최소값
        printf("Warning: nFold value cannot be < 2, nFold value chaged to %d\n", nFold);
    }
    // K가 1보다 작을 경우 실행
    if(K < 1){
        K = 1; // K가 가질 수 있는 최소값
        printf("Warning: K cannot be < 1, K value changed to %d\n", K);
    }

    num_features = 6;
    X = (double*)malloc(num_features * sizeof(double));
    X_prime = (double*)malloc(num_features * sizeof(double));

    for(int curr_fold = 0; curr_fold < nFold; ++curr_fold){
        int testStart, testEnd;
        i = j = 0;

        //테스트 데이터가 포함되는 구간을 계산
        testStart = ((float)N*(curr_fold)/nFold);
        testEnd = ((float)N*(curr_fold+1)/nFold);

        test_size = testEnd - testStart;
        train_size = N-test_size;

        // 거리 계산 결과를 저장할 dists와 훈련데이터, 테스트 데이터들의 주소를 저장하는 P,Q 배열을 선언한다
        dists = (Dist**)malloc(train_size * sizeof(Dist*));
        Q = (SUVData**)malloc(train_size * sizeof(SUVData*));
        P = (SUVData**)malloc(test_size * sizeof(SUVData*));

        // 전체 데이터 중 테스트 데이터와 훈련 데이터로 지정된 데이터의 주소를 저장
        for(int n = 0; n < N; ++n){
            if((n >= testStart) && (n < testEnd)){
                *(P+i) = (*dataset+n);
                ++i;
            }
            else{
                *(Q+j) = (*dataset+n);
                *(dists+j) = (Dist*)malloc(sizeof(Dist));
                ++j;
            }
        }

        // K가 훈련 데이터의 크기보다 클 경우 실행
        if(K > train_size){
            K = train_size; // K가 가질 수 있는 최대값
            printf("Warning: K-value is larger than train size\n> K-value changed to %d\n", K);
        }
        // 아래는 KNN() 함수와 동일하다
        min_age = max_age = (*(Q))->age;
        min_fare = max_fare = (*(Q))->fare;
        min_parch = max_parch = (*(Q))->parch;
        min_sibsp = max_sibsp = (*(Q))->sibsp;
        min_pclass = max_pclass = (*(Q))->parch;
        for(j = 1; j < train_size; ++j){
            if(min_age > (*(Q+j))->age){
                min_age = (*(Q+j))->age;
            }
            if(min_fare > (*(Q+j))->fare){
                min_fare = (*(Q+j))->fare;
            }
            if(min_parch > (*(Q+j))->parch){
                min_parch = (*(Q+j))->parch;
            }
            if(min_sibsp > (*(Q+j))->sibsp){
                min_sibsp = (*(Q+j))->sibsp;
            }
            if(min_pclass > (*(Q+j))->parch){
                min_pclass = (*(Q+j))->parch;
            }
        }

        for(j = 1; j < train_size; ++j){
            if(max_age < (*(Q+j))->age){
                max_age = (*(Q+j))->age;
            }
            if(max_fare < (*(Q+j))->fare){
                max_fare = (*(Q+j))->fare;
            }
            if(max_parch < (*(Q+j))->parch){
                max_parch = (*(Q+j))->parch;
            }
            if(max_sibsp < (*(Q+j))->sibsp){
                max_sibsp = (*(Q+j))->sibsp;
            }
            if(max_pclass < (*(Q+j))->parch){
                max_pclass = (*(Q+j))->parch;
            }
        }

        p = 1;
        num_correct = 0;
        score = 0;
        for(i = 0; i < test_size; ++i){
            survival_state = (*(P+i))->survival;
            *(X) = normalize((*(P+i))->pclass, min_pclass, max_pclass);
            *(X+1) = (*(P+i))->sex;
            *(X+2) = normalize((*(P+i))->age, min_age, max_age);
            *(X+3) = normalize((*(P+i))->sibsp, min_sibsp, max_sibsp);
            *(X+4) = normalize((*(P+i))->parch, min_parch, max_parch);
            *(X+5) = normalize((*(P+i))->fare, min_fare, max_fare);
            for(j = 0; j < train_size; ++j){
                *(X_prime) = normalize((*(Q+j))->pclass, min_pclass, max_pclass);
                *(X_prime+1) = (*(Q+j))->sex;
                *(X_prime+2) = normalize((*(Q+j))->age, min_age, max_age);
                *(X_prime+3) = normalize((*(Q+j))->sibsp, min_sibsp, max_sibsp);
                *(X_prime+4) = normalize((*(Q+j))->parch, min_parch, max_parch);
                *(X_prime+5) = normalize((*(Q+j))->fare, min_fare, max_fare);
                (*(dists+j))->distance = EuclidDist(X, X_prime, num_features);
                (*(dists+j))->index = j;
            }
            for(j = 0; j < train_size-1; ++j){
                for(int w = 0; w < train_size-j-1; ++w){
                    if(((*(dists+w))->distance) > ((*(dists+w+1))->distance)){
                        Dist *temp;
                        temp = *(dists+w);
                        *(dists+w) = *(dists+w+1);
                        *(dists+w+1) = temp;
                    }
                }
            }
            num_survival = num_death = 0;
            for(int k = 0; k < K; ++k){
                curr_idx = (*(dists+k))->index;
                if((*(Q+curr_idx))->survival == 0){
                    num_death++;
                }
                else{
                    num_survival++;
                }
            }
            if(num_death == num_survival){
                curr_idx = (*(dists+K))->index;
                if((*(Q+curr_idx))->survival == 0){
                    num_death++;
                }
                else{
                    num_survival++;
                }
            }
            if(num_survival > num_death){
                if(survival_state == 1)
                    ++num_correct;
            }
            else{
                if(survival_state == 0){
                    ++num_correct;
                }
            }
        }
        score = (float)num_correct/test_size * 100;

        printf("[Fold %d] Prediction Score: %.2f%%\n", curr_fold+1, score);

        for(j = 0; j < train_size; ++j){
            free(*(dists+j));
        }

        free(dists);
        free(P);
        free(Q);
    }
    printf("\n");
    free(X);
    free(X_prime);

    return;
}

// float type의 수를 받아 소수점 한자리 까지만 char string으로 변환하여 리턴하는 함수
char* numToPerString(float number){
    int scaled, num_digit;
    char zero = '0';
    char* str_num = (char*)malloc(6*sizeof(char));

    scaled = number*10; // 소수점 한자리를 끌어올린다
    num_digit = 0;

    for(int p = 1000; p != 0; p/=10){
        if(p == 1){
            *(str_num+num_digit) = '.';
            ++num_digit;
        }
        // 앞자리가 0이 아닐때 string으로의 변환을 시작한다
        if((num_digit != 0) || (scaled/p != 0)){
            *(str_num+num_digit) = zero + scaled/p;
            scaled %= p;
            ++num_digit;
        }
    }

    // 마지막에 '%'와'\0'를 추가
    *(str_num+num_digit) = '%';
    ++num_digit;
    *(str_num+num_digit) = '\0';

    return str_num;
}

// double타입의 수를 받아 정규화하고 리턴하는 함수
double normalize(double num, double min_n, double max_n){
    if((max_n - min_n) != 0){
        num = (num-min_n)/(max_n - min_n);
    }

    return num;
}

// 벡터 X와 X_prime을 받아 두 벡터의 유클리드 거리를 구하고 리턴하는 함수
float EuclidDist(double *X, double *X_prime, int num_features){
    float dist = 0;

    for(int i = 0; i < num_features; ++i){
        dist += pow((*(X+i))-(*(X_prime+i)), 2.0);
    }

    dist = sqrt(dist);

    return dist;
}

// 탑승했던 항구에 따라 생존자의 통계를 내는 함수
void StatsByEmbarked(SUVData** dataset, int N){
    int num_ports, sdSum;
    char *str_ratio, *curr_port;

    num_ports = 3;

    SDRatio *Ports = (SDRatio*)calloc(num_ports, sizeof(SDRatio));

    for(int i = 0; i < N; ++i){
        // 현재 탑승자의 탑승 항구를 확인함
        curr_port = ((*dataset+i)->embarked);
        // 현재 탑승자의 탑승 항구에 따라 해당 항구에서 출발한 사망자 혹은 생존자의 수를 증가
        switch(*curr_port){
            case 'C':
                ((*dataset+i)->survival == 0) ? (Ports+0)->death++ : (Ports+0)->survived++;
                break;
            case 'S':
                ((*dataset+i)->survival == 0) ? (Ports+1)->death++ : (Ports+1)->survived++;
                break;
            default:
                ((*dataset+i)->survival == 0) ? (Ports+2)->death++ : (Ports+2)->survived++;
                break;
        }
    }

    // 통계 출력
    printf("%-15s%s\n", " ", "Embarked");
    printf("%-15s%-15s%-15s%s\n", " ", "Cherbourg", "Southampton", "Queenstown");
    printf("%-15s", "Survived");
    // 생존자의 수 출력
    for(int i = 0; i < num_ports; ++i){
        printf("%-15d", (Ports+i)->survived);
    }
    // 사망자의 수 출력
    printf("\n%-15s", "Dead");
    for(int i = 0; i < num_ports; ++i){
        printf("%-15d", (Ports+i)->death);
    }
    // 생존율 출력
    printf("\n%-15s", "Ratio");
    for(int i = 0; i < num_ports; ++i){
        sdSum = ((Ports+i)->death) + ((Ports+i)->survived);
        if(sdSum == 0){
            (Ports+i)->ratio = 0.0;
        }
        else{
            (Ports+i)->ratio = (float)((Ports+i)->survived)/sdSum * 100;
        }
        str_ratio = numToPerString((Ports+i)->ratio);
        printf("%-15s", str_ratio);
        free(str_ratio);
    }
    printf("\n\n");

    free(Ports);

    return;
}