@rem
@rem Copyright 2015 the original author or authors.
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem      http://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.
@rem

@if "%DEBUG%" == "" @echo off
@rem ##########################################################################
@rem
@rem  browserup-proxy startup script for Windows
@rem
@rem ##########################################################################

@rem Set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" setlocal

set DIRNAME=%~dp0
if "%DIRNAME%" == "" set DIRNAME=.
set APP_BASE_NAME=%~n0
set APP_HOME=%DIRNAME%..

@rem Add default JVM options here. You can also use JAVA_OPTS and BROWSERUP_PROXY_OPTS to pass JVM options to this script.
set DEFAULT_JVM_OPTS=

@rem Find java.exe
if defined JAVA_HOME goto findJavaFromJavaHome

set JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
if "%ERRORLEVEL%" == "0" goto init

echo.
echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.
echo.
echo Please set the JAVA_HOME variable in your environment to match the
echo location of your Java installation.

goto fail

:findJavaFromJavaHome
set JAVA_HOME=%JAVA_HOME:"=%
set JAVA_EXE=%JAVA_HOME%/bin/java.exe

if exist "%JAVA_EXE%" goto init

echo.
echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME%
echo.
echo Please set the JAVA_HOME variable in your environment to match the
echo location of your Java installation.

goto fail

:init
@rem Get command-line arguments, handling Windows variants

if not "%OS%" == "Windows_NT" goto win9xME_args

:win9xME_args
@rem Slurp the command line arguments.
set CMD_LINE_ARGS=
set _SKIP=2

:win9xME_args_slurp
if "x%~1" == "x" goto execute

set CMD_LINE_ARGS=%*

:execute
@rem Setup the command line

set CLASSPATH=%APP_HOME%\lib\browserup-proxy-dist-2.1.0.jar;%APP_HOME%\lib\browserup-proxy-rest-2.1.0.jar;%APP_HOME%\lib\browserup-proxy-core-2.1.0.jar;%APP_HOME%\lib\sitebricks-0.8.11.jar;%APP_HOME%\lib\jersey-media-json-jackson-2.30.1.jar;%APP_HOME%\lib\swagger-jaxrs2-2.1.1.jar;%APP_HOME%\lib\swagger-integration-2.1.1.jar;%APP_HOME%\lib\swagger-core-2.1.1.jar;%APP_HOME%\lib\sitebricks-client-0.8.11.jar;%APP_HOME%\lib\sitebricks-converter-0.8.11.jar;%APP_HOME%\lib\jackson-jaxrs-json-provider-2.10.1.jar;%APP_HOME%\lib\jackson-module-jaxb-annotations-2.10.1.jar;%APP_HOME%\lib\jackson-datatype-jsr310-2.10.1.jar;%APP_HOME%\lib\jackson-jaxrs-base-2.10.1.jar;%APP_HOME%\lib\jackson-databind-2.10.3.jar;%APP_HOME%\lib\swagger-models-2.1.1.jar;%APP_HOME%\lib\jackson-annotations-2.10.3.jar;%APP_HOME%\lib\jackson-dataformat-yaml-2.10.3.jar;%APP_HOME%\lib\jackson-core-2.10.3.jar;%APP_HOME%\lib\guice-servlet-4.2.2.jar;%APP_HOME%\lib\guice-multibindings-4.2.2.jar;%APP_HOME%\lib\guice-4.2.2.jar;%APP_HOME%\lib\log4j-slf4j-impl-2.13.1.jar;%APP_HOME%\lib\log4j-core-2.13.1.jar;%APP_HOME%\lib\log4j-api-2.13.1.jar;%APP_HOME%\lib\jetty-servlet-9.4.26.v20200117.jar;%APP_HOME%\lib\jetty-security-9.4.26.v20200117.jar;%APP_HOME%\lib\jetty-server-9.4.26.v20200117.jar;%APP_HOME%\lib\jersey-hk2-2.30.1.jar;%APP_HOME%\lib\javassist-3.26.0-GA.jar;%APP_HOME%\lib\browserup-proxy-mitm-2.1.0.jar;%APP_HOME%\lib\netty-codec-4.1.44.Final.jar;%APP_HOME%\lib\littleproxy-2.0.0-beta-5.jar;%APP_HOME%\lib\jaxb-api-2.3.1.jar;%APP_HOME%\lib\guava-28.2-jre.jar;%APP_HOME%\lib\jzlib-1.1.3.jar;%APP_HOME%\lib\dnsjava-2.1.9.jar;%APP_HOME%\lib\bcpkix-jdk15on-1.64.jar;%APP_HOME%\lib\bcprov-jdk15on-1.64.jar;%APP_HOME%\lib\dec-0.1.2.jar;%APP_HOME%\lib\selenium-api-3.141.59.jar;%APP_HOME%\lib\jcl-over-slf4j-1.7.30.jar;%APP_HOME%\lib\async-http-client-1.6.3.jar;%APP_HOME%\lib\slf4j-api-1.7.30.jar;%APP_HOME%\lib\jopt-simple-5.0.4.jar;%APP_HOME%\lib\commons-lang3-3.9.jar;%APP_HOME%\lib\jersey-container-servlet-core-2.30.1.jar;%APP_HOME%\lib\jersey-bean-validation-2.30.1.jar;%APP_HOME%\lib\swagger-jaxrs2-servlet-initializer-2.1.1.jar;%APP_HOME%\lib\snakeyaml-1.24.jar;%APP_HOME%\lib\javax.inject-1.jar;%APP_HOME%\lib\aopalliance-1.0.jar;%APP_HOME%\lib\sitebricks-annotations-0.8.11.jar;%APP_HOME%\lib\mvel2-2.1.3.Final.jar;%APP_HOME%\lib\jcip-annotations-1.0.jar;%APP_HOME%\lib\annotations-7.0.3.jar;%APP_HOME%\lib\jsoup-1.8.1.jar;%APP_HOME%\lib\validation-api-1.1.0.Final.jar;%APP_HOME%\lib\javax.servlet-api-3.1.0.jar;%APP_HOME%\lib\jetty-http-9.4.26.v20200117.jar;%APP_HOME%\lib\jetty-io-9.4.26.v20200117.jar;%APP_HOME%\lib\netty-transport-4.1.44.Final.jar;%APP_HOME%\lib\netty-buffer-4.1.44.Final.jar;%APP_HOME%\lib\netty-resolver-4.1.44.Final.jar;%APP_HOME%\lib\netty-common-4.1.44.Final.jar;%APP_HOME%\lib\netty-all-4.1.34.Final.jar;%APP_HOME%\lib\barchart-udt-bundle-2.3.0.jar;%APP_HOME%\lib\javax.activation-api-1.2.0.jar;%APP_HOME%\lib\failureaccess-1.0.1.jar;%APP_HOME%\lib\listenablefuture-9999.0-empty-to-avoid-conflict-with-guava.jar;%APP_HOME%\lib\jsr305-3.0.2.jar;%APP_HOME%\lib\checker-qual-2.10.0.jar;%APP_HOME%\lib\error_prone_annotations-2.3.4.jar;%APP_HOME%\lib\j2objc-annotations-1.3.jar;%APP_HOME%\lib\jersey-server-2.30.1.jar;%APP_HOME%\lib\jersey-client-2.30.1.jar;%APP_HOME%\lib\jersey-media-jaxb-2.30.1.jar;%APP_HOME%\lib\jersey-common-2.30.1.jar;%APP_HOME%\lib\hk2-locator-2.6.1.jar;%APP_HOME%\lib\hk2-api-2.6.1.jar;%APP_HOME%\lib\hk2-utils-2.6.1.jar;%APP_HOME%\lib\jakarta.inject-2.6.1.jar;%APP_HOME%\lib\jersey-entity-filtering-2.30.1.jar;%APP_HOME%\lib\jakarta.ws.rs-api-2.1.6.jar;%APP_HOME%\lib\jakarta.validation-api-2.0.2.jar;%APP_HOME%\lib\hibernate-validator-6.0.17.Final.jar;%APP_HOME%\lib\jakarta.el-api-3.0.3.jar;%APP_HOME%\lib\jakarta.el-3.0.2.jar;%APP_HOME%\lib\classgraph-4.6.32.jar;%APP_HOME%\lib\swagger-annotations-2.1.1.jar;%APP_HOME%\lib\xstream-1.3.1.jar;%APP_HOME%\lib\netty-3.2.4.Final.jar;%APP_HOME%\lib\jetty-util-9.4.26.v20200117.jar;%APP_HOME%\lib\jakarta.annotation-api-1.3.5.jar;%APP_HOME%\lib\osgi-resource-locator-1.0.3.jar;%APP_HOME%\lib\jakarta.xml.bind-api-2.3.2.jar;%APP_HOME%\lib\jakarta.activation-api-1.2.1.jar;%APP_HOME%\lib\aopalliance-repackaged-2.6.1.jar;%APP_HOME%\lib\jboss-logging-3.3.2.Final.jar;%APP_HOME%\lib\classmate-1.3.4.jar;%APP_HOME%\lib\xpp3_min-1.1.4c.jar

@rem Execute browserup-proxy
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %BROWSERUP_PROXY_OPTS%  -classpath "%CLASSPATH%" com.browserup.bup.proxy.Main %CMD_LINE_ARGS%

:end
@rem End local scope for the variables with windows NT shell
if "%ERRORLEVEL%"=="0" goto mainEnd

:fail
rem Set variable BROWSERUP_PROXY_EXIT_CONSOLE if you need the _script_ return code instead of
rem the _cmd.exe /c_ return code!
if  not "" == "%BROWSERUP_PROXY_EXIT_CONSOLE%" exit 1
exit /b 1

:mainEnd
if "%OS%"=="Windows_NT" endlocal

:omega
