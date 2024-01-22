#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <algorithm>
#include <functional>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shaderprogram.h"
#include "camera.h"
#include "util.h"

constexpr GLuint WIDTH            = 1280;
constexpr GLuint HEIGHT           = 720;
constexpr GLuint BUFFER_SIZE      = WIDTH * HEIGHT * sizeof(GLfloat);
constexpr GLfloat UPDATE_TIME     = 0.016F;

GLvoid error_callback(GLint error, const GLchar* description);
GLvoid createModel(GLuint& vao, GLuint& vbo, GLuint& ebo);
GLuint createTextureObject();

template<typename T>
class ShaderBuffer
{
public:
    ShaderBuffer(GLuint name, GLsizei size) :
        mName(name),
        mSize(size)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, name);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * size, nullptr,
                     GL_DYNAMIC_COPY);
    }

    GLuint name() { return mName; }

    void initialize(std::function<T (T&)> f)
    {
        // Get a pointer to each buffer so we can initialize its contents
        glBindBuffer(GL_ARRAY_BUFFER, mName);
        GLfloat* p = reinterpret_cast<GLfloat*>(
            glMapBufferRange(GL_ARRAY_BUFFER, 0, sizeof(T) * mSize,
                             GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));

        std::transform(p, p + mSize, p, f);

        glUnmapBuffer(GL_ARRAY_BUFFER);
    }

    void bind(GLuint index)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, mName);
    }

private:
    GLuint mName;
    GLsizei mSize;
};

int main()
{
    glfwSetErrorCallback(error_callback);

    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL App",
                                          nullptr, nullptr);

    if(!window)
    {
        glfwTerminate();
        exit(1);
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    GLenum status = glewInit();

    if(status != GLEW_OK)
    {
        std::cerr << "GLEW error: " << glewGetErrorString(status) << "\n";

        glfwTerminate();
        exit(1);
    }

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(simgll::debug_callback, nullptr);

    // Create render and compute programs
    simgll::ShaderProgram renderProgram;
    renderProgram.addShader("vertex_shader.glsl",   GL_VERTEX_SHADER);
    renderProgram.addShader("fragment_shader.glsl", GL_FRAGMENT_SHADER);
    renderProgram.compile();

    simgll::ShaderProgram computeProgram;
    computeProgram.addShader("compute_shader.glsl", GL_COMPUTE_SHADER);
    computeProgram.compile();

    // Initialize PRNG
    std::random_device rd;
    unsigned seed;

    // Check if the implementation provides a usable random_device
    if (0 != rd.entropy())
    {
        seed = rd();
    }
    else
    {
        // No random device available, seed using the system clock
        seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    }

    std::mt19937 engine(seed);
    std::uniform_real_distribution<> dist(0.0F, 1.0F);

    // Create input / output buffers
    GLuint buffers[4];
    glGenBuffers(4, buffers);

    for(int i = 0; i < 4; i++)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, BUFFER_SIZE, nullptr,
                     GL_DYNAMIC_COPY);
    }

    ShaderBuffer<GLfloat> a1(buffers[0], WIDTH * HEIGHT);
    a1.initialize([](GLfloat&) { return 1.0F; });

    ShaderBuffer<GLfloat> a2(buffers[1], WIDTH * HEIGHT);
    a2.initialize([](GLfloat&) { return 1.0F; });

    ShaderBuffer<GLfloat> b1(buffers[2], WIDTH * HEIGHT);
    b1.initialize([&dist, &engine](GLfloat&) {
                   if (dist(engine) < 0.000021) return 1.0F;
                   return 0.0F;
                   });

    ShaderBuffer<GLfloat> b2(buffers[3], WIDTH * HEIGHT);
    b2.initialize([](GLfloat&) { return 0.0f; });

    // Create quad
    GLuint vao, vbo, ebo;
    createModel(vao, vbo, ebo);

    // Create output texture
    GLuint outputTexture = createTextureObject();

    glViewport(0, 0, WIDTH, HEIGHT);
    glClearColor(0.0F, 0.0F, 0.0F, 1.0F);

    GLfloat currentTime = 0.0F;
    GLfloat oldTime     = 0.0F;
    GLfloat deltaTime   = 0.0F;
    GLfloat totalTime   = 0.0f;

    while(!glfwWindowShouldClose(window))
    {
        currentTime = (GLfloat)glfwGetTime();
        deltaTime   = currentTime - oldTime;
        oldTime     = currentTime;
        totalTime  += deltaTime;

        if (totalTime >= UPDATE_TIME)
        {
            // Bind buffers for compute shader
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[0]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers[2]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, buffers[3]);
            glBindImageTexture(4, outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

            computeProgram.use();

            glDispatchCompute(WIDTH / 20, HEIGHT / 20, 1);

            // To ensure the visibility of writes to the outputTexture between
            // shader invocations in two different rendering commands, i.e.,
            // making an incoherent write from one command visible to a read in
            // a later OpenGL command (external visibility), we must use a call
            // to the follwing function between the writing OpenGL call and the
            // reading OpenGL call. The thing to keep in mind about the various
            // bits in the bitfield is this: they represent the operation you
            // want to make the incoherent memory access visible to. If you
            // want to image load/store operations from one command to be
            // visible to image load/store operations from another command you
            // must use GL_SHADER_IMAGE_ACCESS_BARRIER_BIT.
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            std::swap(buffers[0], buffers[1]);
            std::swap(buffers[2], buffers[3]);

            totalTime = 0.0F;
        }

        // Rendering
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT);

        renderProgram.use();

        glBindTextureUnit(0, outputTexture);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (GLvoid*)0);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &outputTexture);

    glfwTerminate();

    return 0;
}

GLvoid error_callback(GLint error, const GLchar* description)
{
    std::cerr << "GLFW error " << error << ": " << description << "\n";
    exit(1);
}

GLuint createTextureObject()
{
    GLuint texture;

    glCreateTextures(GL_TEXTURE_2D, 1, &texture);

    glTextureStorage2D(texture, 1, GL_RGBA8, WIDTH, HEIGHT);

    glTextureParameteri(texture, GL_TEXTURE_WRAP_S,     GL_REPEAT);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_T,     GL_REPEAT);
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    return texture;
}

void createModel(GLuint& vao, GLuint& vbo, GLuint& ebo)
{
    GLfloat vertices[] =
    {
        -1.0f, -1.0f, 0.0f,     0.0f, 0.0f,
         1.0f, -1.0f, 0.0f,     1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,     1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f,     0.0f, 1.0f,
    };

    GLuint elements[] =
    {
        0, 1, 3,
        1, 2, 3
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(0 * sizeof(GLfloat)));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}

